/**
 * Expose a record's keys as properties of `target` without letting them shadow
 * anything `target` already has.
 *
 * `Example` and `Prediction` want `ex.question` to work, but a field named
 * `get`, `toObject` or `trace` must not replace the method or property of the
 * same name — defining data keys as own properties did exactly that. The proxy
 * resolves names against the instance (and its prototype chain) first and only
 * then against the data, so methods always win while the data stays reachable
 * through `get(key)` and `toObject()`.
 *
 * Reflection reports the data: `Object.keys`, spreading and `for...in` see the
 * data keys that are reachable as properties, not the instance's internals.
 */
export function withDataProperties<T extends object>(
    target: T,
    data: () => Record<string, any>,
    options: { writable: boolean }
): T {
    const own = (key: PropertyKey): boolean =>
        Object.prototype.hasOwnProperty.call(target, key);
    /** A data key that is visible as a property: present, and not shadowed. */
    const isDataKey = (key: PropertyKey): key is string =>
        typeof key === 'string' &&
        !(key in target) &&
        Object.prototype.hasOwnProperty.call(data(), key);

    return new Proxy(target, {
        get(t, key, receiver) {
            if (typeof key === 'symbol' || key in t) return Reflect.get(t, key, receiver);
            return isDataKey(key) ? data()[key] : undefined;
        },
        set(t, key, value) {
            if (typeof key === 'symbol' || own(key)) return Reflect.set(t, key, value);
            if (!options.writable) return false;
            data()[key] = value;
            return true;
        },
        deleteProperty(t, key) {
            if (typeof key === 'symbol' || own(key)) return Reflect.deleteProperty(t, key);
            if (!options.writable) return false;
            delete data()[key];
            return true;
        },
        has(t, key) {
            return key in t || isDataKey(key);
        },
        ownKeys() {
            return Object.keys(data()).filter(isDataKey);
        },
        getOwnPropertyDescriptor(t, key) {
            if (isDataKey(key)) {
                return {
                    value: data()[key],
                    writable: options.writable,
                    enumerable: true,
                    configurable: true,
                };
            }
            return Reflect.getOwnPropertyDescriptor(t, key);
        },
    });
}
