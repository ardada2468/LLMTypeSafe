/**
 * `RespAct` was renamed {@link ReAct}, matching DSPy. These aliases keep old
 * imports compiling; they will be removed in a future major release.
 */
import { ReAct, type ReActEvent, type ReActOptions } from './react';
import { type SignatureSource } from '../core/signature';

/** @deprecated Renamed `ReAct`. */
export const RespAct = ReAct;
/** @deprecated Renamed `ReAct`. */
export type RespAct<TSignature extends SignatureSource = any> = ReAct<TSignature>;
/** @deprecated Renamed `ReActEvent`. */
export type RespActEvent = ReActEvent;
/** @deprecated Renamed `ReActOptions`. */
export type RespActOptions = ReActOptions;
