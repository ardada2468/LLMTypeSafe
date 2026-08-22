import { createRng, mapWithConcurrency, shuffled } from './random';

describe('createRng', () => {
    it('produces the same sequence for the same seed', () => {
        const first = Array.from({ length: 5 }, createRng(7));
        const second = Array.from({ length: 5 }, createRng(7));

        expect(first).toEqual(second);
    });

    it('produces a different sequence for a different seed', () => {
        expect(createRng(1)()).not.toBe(createRng(2)());
    });

    it('stays within [0, 1)', () => {
        const rng = createRng(-13);

        for (let i = 0; i < 500; i++) {
            const value = rng();
            expect(value).toBeGreaterThanOrEqual(0);
            expect(value).toBeLessThan(1);
        }
    });
});

describe('shuffled', () => {
    it('leaves the input array alone', () => {
        const items = [1, 2, 3, 4, 5];

        shuffled(items, createRng(3));

        expect(items).toEqual([1, 2, 3, 4, 5]);
    });

    it('keeps every element exactly once', () => {
        const items = [1, 2, 3, 4, 5, 6, 7, 8];

        const result = shuffled(items, createRng(11));

        expect([...result].sort((a, b) => a - b)).toEqual(items);
    });

    it('orders identically for the same seed', () => {
        const items = ['a', 'b', 'c', 'd', 'e', 'f'];

        expect(shuffled(items, createRng(5))).toEqual(shuffled(items, createRng(5)));
    });
});

describe('mapWithConcurrency', () => {
    it('returns results in input order, not completion order', async () => {
        const delays = [30, 1, 20, 2];

        const results = await mapWithConcurrency(delays, 4, async (delay, index) => {
            await new Promise((resolve) => setTimeout(resolve, delay));
            return index;
        });

        expect(results).toEqual([0, 1, 2, 3]);
    });

    it('runs no more than the limit at once', async () => {
        let inFlight = 0;
        let peak = 0;

        await mapWithConcurrency(
            Array.from({ length: 10 }, (_, i) => i),
            3,
            async () => {
                inFlight += 1;
                peak = Math.max(peak, inFlight);
                await new Promise((resolve) => setTimeout(resolve, 1));
                inFlight -= 1;
                return null;
            }
        );

        expect(peak).toBe(3);
    });

    it('handles an empty list', async () => {
        expect(await mapWithConcurrency([], 4, async () => 1)).toEqual([]);
    });

    it('treats a limit below one as one', async () => {
        const order: number[] = [];

        await mapWithConcurrency([1, 2, 3], 0, async (item) => {
            order.push(item);
            await new Promise((resolve) => setTimeout(resolve, 1));
            return item;
        });

        expect(order).toEqual([1, 2, 3]);
    });
});
