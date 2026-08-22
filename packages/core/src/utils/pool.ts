/**
 * A bounded worker pool.
 *
 * Every non-trivial program built on this library eventually grows a
 * hand-rolled loop over a list of inputs, and the detail those loops get wrong
 * is ordering: results come back in completion order rather than input order.
 * This primitive keeps the two separate — work is claimed in order and results
 * are written back by index, so slot `i` always holds the result for item `i`.
 */

/** Per-item outcome, mirroring the shape of `Promise.allSettled`. */
export type SettledResult<T> =
    { status: 'fulfilled'; value: T } | { status: 'rejected'; reason: unknown };

export interface MapWithConcurrencyOptions {
    /** Maximum number of items in flight at once. Must be a positive integer. */
    concurrency?: number;
    /**
     * Called as each item settles, whether it fulfilled or rejected. Advisory
     * only: if it throws it is dropped for the rest of the run, which carries on.
     */
    onSettled?: (done: number, total: number) => void;
    /**
     * Reject the whole run on the first item failure instead of capturing it.
     * In-flight items are still awaited and no further items are started; the
     * run rejects with the lowest-indexed failure among those observed.
     */
    stopOnError?: boolean;
    /** Stop starting new items once aborted, then reject with the abort reason. */
    signal?: AbortSignal;
}

/** Default in-flight limit: high enough to matter, low enough to stay polite. */
export const DEFAULT_CONCURRENCY = 8;

/** Reject a concurrency value that could never make progress. */
export function assertValidConcurrency(concurrency: number): void {
    if (!Number.isInteger(concurrency) || concurrency < 1) {
        throw new RangeError(
            `concurrency must be a positive integer, received ${String(concurrency)}`
        );
    }
}

function abortReason(signal: AbortSignal): unknown {
    return signal.reason ?? new Error('The operation was aborted');
}

/**
 * Run `worker` over `items` with at most `concurrency` calls in flight.
 *
 * Resolves to one {@link SettledResult} per input, **in input order**, however
 * out of order the individual promises settled. A failing item does not stop
 * the run unless `stopOnError` is set — one bad row should not destroy a
 * ten-thousand-row job.
 *
 * These are the only two ways the run is abandoned. In particular an
 * `onSettled` that throws is the caller's bug, not the batch's: progress
 * reporting is advisory, so the callback is dropped for the rest of the run and
 * the work still completes.
 *
 * @throws RangeError if `concurrency` is not a positive integer.
 * @throws the lowest-indexed item's failure reason when `stopOnError` is set.
 * @throws the signal's abort reason when `signal` aborts, or has already aborted.
 */
export async function mapWithConcurrency<T, R>(
    items: readonly T[],
    worker: (item: T, index: number) => Promise<R> | R,
    options: MapWithConcurrencyOptions = {}
): Promise<SettledResult<R>[]> {
    const {
        concurrency = DEFAULT_CONCURRENCY,
        onSettled,
        stopOnError = false,
        signal,
    } = options;

    assertValidConcurrency(concurrency);

    if (signal?.aborted) {
        throw abortReason(signal);
    }

    const total = items.length;
    const results: SettledResult<R>[] = new Array(total);
    if (total === 0) {
        return results;
    }

    let nextIndex = 0;
    let settled = 0;
    // Set once `stopOnError` has seen a failure: workers wind down on their own
    // rather than the run rejecting out from under them, so nothing is left
    // running unobserved after this function returns.
    let stopped = false;
    // The failure the run will reject with. Items already in flight keep settling
    // after the first one, and a later-started item can fail sooner, so keep the
    // lowest-indexed failure: a re-run against the same flaky data then reports
    // the same row rather than whichever lost the race.
    let fatal: { reason: unknown; index: number } | undefined;
    // Dropped the first time it throws. A progress bar writing to a closed
    // stream must not cost the caller ten thousand completed rows.
    let reportProgress = onSettled;

    const runWorker = async (): Promise<void> => {
        while (!stopped && signal?.aborted !== true) {
            const index = nextIndex;
            if (index >= total) {
                return;
            }
            nextIndex += 1;

            try {
                results[index] = {
                    status: 'fulfilled',
                    value: await worker(items[index] as T, index),
                };
            } catch (reason) {
                results[index] = { status: 'rejected', reason };
                if (stopOnError) {
                    stopped = true;
                    if (!fatal || index < fatal.index) {
                        fatal = { reason, index };
                    }
                }
            }

            settled += 1;
            try {
                reportProgress?.(settled, total);
            } catch {
                reportProgress = undefined;
            }
        }
    };

    const workerCount = Math.min(concurrency, total);
    await Promise.all(Array.from({ length: workerCount }, () => runWorker()));

    if (fatal !== undefined) {
        throw fatal.reason;
    }
    if (signal?.aborted) {
        throw abortReason(signal);
    }

    return results;
}
