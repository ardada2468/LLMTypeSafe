import { createRng, shuffled } from './random';

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
