import { mapWithConcurrency, assertValidConcurrency, DEFAULT_CONCURRENCY } from './pool';

/** A promise plus the handles to settle it, so a test controls completion order. */
function deferred<T>() {
    let resolve!: (value: T) => void;
    let reject!: (reason: unknown) => void;
    const promise = new Promise<T>((res, rej) => {
        resolve = res;
        reject = rej;
    });
    return { promise, resolve, reject };
}

const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

/** Records how many calls are in flight at once. */
function trackingWorker(delayFor: (index: number) => number) {
    const state = { inFlight: 0, peak: 0 };
    const worker = async (item: number, index: number) => {
        state.inFlight += 1;
        state.peak = Math.max(state.peak, state.inFlight);
        await new Promise((resolve) => setTimeout(resolve, delayFor(index)));
        state.inFlight -= 1;
        return item * 2;
    };
    return { state, worker };
}

describe('mapWithConcurrency', () => {
    it('returns results in input order when items settle out of order', async () => {
        const gates = [deferred<string>(), deferred<string>(), deferred<string>()];

        const run = mapWithConcurrency([0, 1, 2], (item: number) => gates[item]!.promise, {
            concurrency: 3,
        });
        await tick();
        gates[2]!.resolve('third');
        gates[0]!.resolve('first');
        gates[1]!.resolve('second');

        expect(await run).toEqual([
            { status: 'fulfilled', value: 'first' },
            { status: 'fulfilled', value: 'second' },
            { status: 'fulfilled', value: 'third' },
        ]);
    });

    it.each([1, 2, 3, 5, 8])('never exceeds a concurrency of %i in flight', async (limit) => {
        const items = Array.from({ length: 20 }, (_, index) => index);
        // Descending delays: later items finish first, so slots free out of order.
        const { state, worker } = trackingWorker((index) => (20 - index) % 5);

        const results = await mapWithConcurrency(items, worker, { concurrency: limit });

        expect(state.peak).toBeLessThanOrEqual(limit);
        expect(state.inFlight).toBe(0);
        expect(results.map((result) => result.status === 'fulfilled' && result.value)).toEqual(
            items.map((item) => item * 2)
        );
    });

    it('runs a single item at a time at a concurrency of 1', async () => {
        const { state, worker } = trackingWorker(() => 1);

        await mapWithConcurrency([0, 1, 2, 3], worker, { concurrency: 1 });

        expect(state.peak).toBe(1);
    });

    it('starts DEFAULT_CONCURRENCY items when no limit is given', async () => {
        const { state, worker } = trackingWorker(() => 1);

        await mapWithConcurrency(
            Array.from({ length: 30 }, (_, i) => i),
            worker
        );

        expect(state.peak).toBe(DEFAULT_CONCURRENCY);
    });

    it('captures a per-item failure without failing its neighbours', async () => {
        const boom = new Error('item 1 failed');

        const results = await mapWithConcurrency([0, 1, 2], async (item: number) => {
            if (item === 1) throw boom;
            return item;
        });

        expect(results).toEqual([
            { status: 'fulfilled', value: 0 },
            { status: 'rejected', reason: boom },
            { status: 'fulfilled', value: 2 },
        ]);
    });

    it('rejects with the first failure when stopOnError is set', async () => {
        const boom = new Error('nope');
        let started = 0;

        const run = mapWithConcurrency(
            Array.from({ length: 20 }, (_, i) => i),
            async (item: number) => {
                started += 1;
                if (item === 0) throw boom;
                await tick();
                return item;
            },
            { concurrency: 2, stopOnError: true }
        );

        await expect(run).rejects.toBe(boom);
        expect(started).toBeLessThan(20);
    });

    it('reports progress as items settle, including failures', async () => {
        const progress: Array<[number, number]> = [];

        await mapWithConcurrency(
            [0, 1, 2, 3],
            async (item: number) => {
                if (item === 2) throw new Error('x');
                return item;
            },
            { concurrency: 2, onSettled: (done, total) => progress.push([done, total]) }
        );

        expect(progress).toEqual([
            [1, 4],
            [2, 4],
            [3, 4],
            [4, 4],
        ]);
    });

    it('stops starting new items once the signal aborts', async () => {
        const controller = new AbortController();
        const started: number[] = [];

        const run = mapWithConcurrency(
            Array.from({ length: 20 }, (_, i) => i),
            async (item: number) => {
                started.push(item);
                if (item === 1) controller.abort();
                await tick();
                return item;
            },
            { concurrency: 2, signal: controller.signal }
        );

        await expect(run).rejects.toThrow();
        expect(started.length).toBeLessThan(20);
    });

    it('rejects immediately when the signal is already aborted', async () => {
        const controller = new AbortController();
        controller.abort(new Error('too late'));
        const worker = vi.fn(async (item: number) => item);

        await expect(
            mapWithConcurrency([0, 1], worker, { signal: controller.signal })
        ).rejects.toThrow('too late');

        expect(worker).not.toHaveBeenCalled();
    });

    it('resolves to an empty array for an empty input list', async () => {
        const worker = vi.fn(async (item: number) => item);

        expect(await mapWithConcurrency([] as number[], worker)).toEqual([]);
        expect(worker).not.toHaveBeenCalled();
    });

    it('finishes the run when onSettled throws, dropping the callback instead', async () => {
        let invocations = 0;

        const results = await mapWithConcurrency([0, 1, 2, 3], async (item: number) => item, {
            concurrency: 2,
            onSettled: () => {
                invocations += 1;
                throw new Error('progress handler blew up');
            },
        });

        expect(results.map((result) => result.status === 'fulfilled' && result.value)).toEqual([
            0, 1, 2, 3,
        ]);
        expect(invocations).toBe(1);
    });

    it('rejects with the lowest-indexed failure when several fail under stopOnError', async () => {
        const first = new Error('index 0');
        const second = new Error('index 1');

        const run = mapWithConcurrency(
            [0, 1, 2, 3],
            async (item: number) => {
                // Index 1 fails immediately; index 0 fails later, but wins.
                if (item === 1) throw second;
                if (item === 0) {
                    await tick();
                    throw first;
                }
                return item;
            },
            { concurrency: 2, stopOnError: true }
        );

        await expect(run).rejects.toBe(first);
    });

    it('rejects an empty input list when the signal has already aborted', async () => {
        const controller = new AbortController();
        controller.abort(new Error('too late'));

        await expect(
            mapWithConcurrency([] as number[], async (item: number) => item, {
                signal: controller.signal,
            })
        ).rejects.toThrow('too late');
    });

    it.each([0, -1, -8, 1.5, NaN, Infinity])(
        'throws RangeError rather than hanging on a concurrency of %s',
        async (concurrency) => {
            const worker = vi.fn(async (item: number) => item);

            await expect(mapWithConcurrency([1, 2], worker, { concurrency })).rejects.toThrow(
                RangeError
            );

            expect(worker).not.toHaveBeenCalled();
        }
    );

    it('accepts a synchronous worker', async () => {
        const results = await mapWithConcurrency([1, 2, 3], (item: number) => item + 1);

        expect(results.map((result) => result.status === 'fulfilled' && result.value)).toEqual([
            2, 3, 4,
        ]);
    });
});

describe('assertValidConcurrency', () => {
    it('accepts any positive integer', () => {
        expect(() => assertValidConcurrency(1)).not.toThrow();
        expect(() => assertValidConcurrency(1000)).not.toThrow();
    });

    it('names the offending value in the message', () => {
        expect(() => assertValidConcurrency(0)).toThrow('received 0');
    });
});
