import { TsDspyError } from '../core/errors';
import type {
    ContentPart,
    ImageContentPart,
    ImageInput,
    ImageMediaType,
    ImageSource,
    MessageContent,
    TextContentPart,
} from '../types/language-model';

/**
 * `data:image/png;base64,iVBOR…` — media type, any further parameters, payload.
 * The middle group exists so a URI carrying e.g. `;name=sign.png` still parses
 * rather than being mistaken for something that is not a data URI at all.
 */
const DATA_URI = /^data:([^;,]+)(;[^,]*)?;base64,([\s\S]*)$/;

/** Build a text part. Mostly sugar, but it keeps call sites symmetrical. */
export function textPart(text: string): TextContentPart {
    return { type: 'text', text };
}

/**
 * Normalise anything a caller may reasonably hand us into an image part.
 *
 * Accepts an `https://` URL, a `data:` URI, an explicit {@link ImageSource}, or
 * an already-built part. A bare base64 blob is rejected: without a media type
 * no provider can be told what it is, and guessing from the payload's first
 * bytes is the kind of cleverness that fails silently in production.
 */
export function imagePart(
    image: ImageInput,
    detail?: ImageContentPart['detail']
): ImageContentPart {
    const part = toImagePart(image);
    return detail === undefined ? part : { ...part, detail };
}

function toImagePart(image: ImageInput): ImageContentPart {
    if (typeof image === 'string') {
        return { type: 'image', source: sourceFromString(image) };
    }
    // Inputs arrive from user-supplied records, so a number or null can reach
    // here; say what was expected rather than failing inside `in`.
    if (typeof image !== 'object' || image === null) {
        throw new TsDspyError(
            `Expected an image URL, a data URI, or an image source object, got ${typeof image}.`
        );
    }
    // `kind` is the source discriminant; a part has `type: 'image'` instead.
    if ('kind' in image) {
        return { type: 'image', source: image };
    }
    return image;
}

function sourceFromString(value: string): ImageSource {
    const dataUri = value.match(DATA_URI);
    if (dataUri) {
        const mediaType = dataUri[1];
        if (!mediaType.startsWith('image/')) {
            throw new TsDspyError(
                `"${mediaType}" is not an image media type. Only images can be sent as ` +
                    'image content; a provider would reject anything else.'
            );
        }
        return { kind: 'base64', mediaType, data: dataUri[3] };
    }
    if (/^https?:\/\//i.test(value)) {
        return { kind: 'url', url: value };
    }
    throw new TsDspyError(
        'Image strings must be an http(s) URL or a "data:<media-type>;base64,…" URI. ' +
            'For raw base64, pass { kind: "base64", data, mediaType } so the media type is known.'
    );
}

/**
 * Rewrite a `data:` URI carried in a URL source as a base64 source.
 *
 * Providers that take inline bytes (Anthropic, Gemini) need the media type and
 * payload separately, and a caller is free to hand us a data URI either way
 * round. Anything else passes through untouched.
 */
export function normalizeImageSource(source: ImageSource): ImageSource {
    if (source.kind !== 'url') return source;
    const dataUri = source.url.match(DATA_URI);
    if (!dataUri) return source;
    return { kind: 'base64', mediaType: dataUri[1], data: dataUri[3] };
}

export function isImagePart(part: ContentPart): part is ImageContentPart {
    return part.type === 'image';
}

/** Render an image source as a `data:` URI, or pass a URL straight through. */
export function imageToUrl(source: ImageSource): string {
    return source.kind === 'url'
        ? source.url
        : `data:${source.mediaType};base64,${source.data}`;
}

/**
 * Media type of an image source, inferred from a `data:` URI when the source is
 * a URL that carries one. `undefined` when it genuinely cannot be known.
 */
export function imageMediaType(source: ImageSource): ImageMediaType | undefined {
    if (source.kind === 'base64') return source.mediaType;
    if (source.mediaType) return source.mediaType;
    return source.url.match(DATA_URI)?.[1];
}

/** Content as an array of parts, wrapping a plain string in a single text part. */
export function toContentParts(content: MessageContent): ContentPart[] {
    return typeof content === 'string' ? [textPart(content)] : content;
}

/** True when the content carries at least one image. */
export function hasImageContent(content: MessageContent): boolean {
    return typeof content !== 'string' && content.some(isImagePart);
}

/**
 * Flatten content to text, replacing images with a short placeholder.
 *
 * Used wherever a channel cannot carry an image at all — Anthropic's `system`
 * parameter, OpenAI's system and assistant roles — so an image degrades to a
 * visible marker rather than `[object Object]`.
 */
export function contentToText(content: MessageContent): string {
    if (typeof content === 'string') return content;
    return content.map((part) => (isImagePart(part) ? placeholder(part) : part.text)).join('');
}

function placeholder(part: ImageContentPart): string {
    const mediaType = imageMediaType(part.source);
    return mediaType ? `[image: ${mediaType}]` : '[image]';
}
