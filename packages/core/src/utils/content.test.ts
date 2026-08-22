import { TsDspyError } from '../core/errors';
import {
    contentToText,
    hasImageContent,
    imageMediaType,
    imagePart,
    imageToUrl,
    isImagePart,
    normalizeImageSource,
    textPart,
    toContentParts,
} from './content';

const PNG = 'iVBORw0KGgo=';
const DATA_URI = `data:image/png;base64,${PNG}`;

describe('imagePart', () => {
    it('splits a data URI into its media type and payload', () => {
        expect(imagePart(DATA_URI)).toEqual({
            type: 'image',
            source: { kind: 'base64', mediaType: 'image/png', data: PNG },
        });
    });

    it('treats an http(s) string as a URL source', () => {
        expect(imagePart('https://example.com/cat.jpg')).toEqual({
            type: 'image',
            source: { kind: 'url', url: 'https://example.com/cat.jpg' },
        });
    });

    it('rejects a bare base64 blob, which carries no media type', () => {
        expect(() => imagePart(PNG)).toThrow(TsDspyError);
        expect(() => imagePart(PNG)).toThrow(/media type/);
    });

    it('rejects a data URI that is not an image', () => {
        expect(() => imagePart('data:application/pdf;base64,JVBERi0=')).toThrow(
            /not an image media type/
        );
    });

    it('parses a data URI carrying extra parameters', () => {
        expect(imagePart(`data:image/jpeg;name=sign.jpg;base64,${PNG}`).source).toEqual({
            kind: 'base64',
            mediaType: 'image/jpeg',
            data: PNG,
        });
    });

    it('says what it expected when handed something that is not an image at all', () => {
        expect(() => imagePart(42 as never)).toThrow(TsDspyError);
        expect(() => imagePart(null as never)).toThrow(/Expected an image URL/);
    });

    it('wraps an explicit source', () => {
        expect(imagePart({ kind: 'base64', data: PNG, mediaType: 'image/webp' })).toEqual({
            type: 'image',
            source: { kind: 'base64', data: PNG, mediaType: 'image/webp' },
        });
    });

    it('returns a ready-made part unchanged', () => {
        const part = imagePart(DATA_URI);
        expect(imagePart(part)).toBe(part);
    });

    it('applies a detail hint without mutating the original part', () => {
        const part = imagePart(DATA_URI);
        expect(imagePart(part, 'low').detail).toBe('low');
        expect(part.detail).toBeUndefined();
    });
});

describe('image sources', () => {
    it('renders a base64 source as a data URI', () => {
        expect(imageToUrl({ kind: 'base64', data: PNG, mediaType: 'image/png' })).toBe(
            DATA_URI
        );
    });

    it('passes a URL through untouched', () => {
        expect(imageToUrl({ kind: 'url', url: 'https://example.com/a.png' })).toBe(
            'https://example.com/a.png'
        );
    });

    it('rewrites a data URI given as a URL source into base64 form', () => {
        expect(normalizeImageSource({ kind: 'url', url: DATA_URI })).toEqual({
            kind: 'base64',
            mediaType: 'image/png',
            data: PNG,
        });
    });

    it('leaves a real URL alone', () => {
        const source = { kind: 'url' as const, url: 'https://example.com/a.png' };
        expect(normalizeImageSource(source)).toBe(source);
    });

    it('reads the media type out of a data URI, and reports none for a bare URL', () => {
        expect(imageMediaType({ kind: 'url', url: DATA_URI })).toBe('image/png');
        expect(imageMediaType({ kind: 'url', url: 'https://example.com/a' })).toBeUndefined();
    });
});

describe('content helpers', () => {
    it('wraps a string in a single text part', () => {
        expect(toContentParts('hi')).toEqual([{ type: 'text', text: 'hi' }]);
    });

    it('reports whether content carries an image', () => {
        expect(hasImageContent('hi')).toBe(false);
        expect(hasImageContent([textPart('hi')])).toBe(false);
        expect(hasImageContent([textPart('hi'), imagePart(DATA_URI)])).toBe(true);
    });

    it('identifies image parts', () => {
        expect(isImagePart(imagePart(DATA_URI))).toBe(true);
        expect(isImagePart(textPart('hi'))).toBe(false);
    });

    it('flattens parts to text, standing an image in for its pixels', () => {
        const flattened = contentToText([
            textPart('look: '),
            imagePart(DATA_URI),
            textPart('\nwhat is it?'),
        ]);

        expect(flattened).toBe('look: [image: image/png]\nwhat is it?');
    });

    it('falls back to a bare placeholder when the media type is unknown', () => {
        expect(contentToText([imagePart('https://example.com/a')])).toBe('[image]');
    });

    it('returns a string content unchanged', () => {
        expect(contentToText('plain')).toBe('plain');
    });
});
