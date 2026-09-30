// PDF text extraction and page rendering for mlx-serve's `document` blocks
// (src/pdf.zig), via PDFKit and CoreGraphics. ARC objc; macOS only (the Linux /
// iOS builds compile src/pdf.zig's stub branch instead).
#import <Foundation/Foundation.h>
#import <PDFKit/PDFKit.h>
#import <CoreGraphics/CoreGraphics.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// The document's text, one "--- page N ---" section per page, as a malloc'd
// UTF-8 string of *len_out bytes (free with mlxs_pdf_free). NULL when the bytes
// are not a PDF PDFKit can open, or it is encrypted with a non-empty password;
// *pages_out is still set when the page count is known. *text_pages_out counts
// the pages that had any text (0 for a scanned document), and *flags_out (when
// non-NULL on success) is a malloc'd array of *pages_out bytes, 1 where that
// page has text (free with mlxs_pdf_free).
char *mlxs_pdf_text(const unsigned char *data, size_t len, int *pages_out, int *text_pages_out, size_t *len_out, unsigned char **flags_out) {
    *pages_out = 0;
    *text_pages_out = 0;
    *len_out = 0;
    *flags_out = NULL;
    @autoreleasepool {
        NSData *bytes = [NSData dataWithBytesNoCopy:(void *)data length:len freeWhenDone:NO];
        PDFDocument *doc = [[PDFDocument alloc] initWithData:bytes];
        if (doc == nil) return NULL;
        *pages_out = (int)doc.pageCount;
        if (doc.isLocked && ![doc unlockWithPassword:@""]) return NULL;
        NSMutableString *out = [NSMutableString string];
        NSCharacterSet *ws = [NSCharacterSet whitespaceAndNewlineCharacterSet];
        NSInteger n = doc.pageCount;
        unsigned char *flags = malloc(n > 0 ? (size_t)n : 1);
        if (flags == NULL) return NULL;
        int with_text = 0;
        for (NSInteger i = 0; i < n; i++) {
            PDFPage *page = [doc pageAtIndex:i];
            NSString *s = [page.string stringByTrimmingCharactersInSet:ws] ?: @"";
            if (i > 0) [out appendString:@"\n\n"];
            [out appendFormat:@"--- page %ld ---", (long)(i + 1)];
            flags[i] = s.length > 0 ? 1 : 0;
            if (s.length > 0) {
                with_text++;
                [out appendString:@"\n"];
                [out appendString:s];
            }
        }
        const char *utf8 = out.UTF8String;
        size_t ulen = utf8 ? strlen(utf8) : 0;
        char *res = utf8 ? malloc(ulen + 1) : NULL;
        if (res == NULL) {
            free(flags);
            return NULL;
        }
        memcpy(res, utf8, ulen + 1);
        *text_pages_out = with_text;
        *len_out = ulen;
        *flags_out = flags;
        return res;
    }
}

// Page `index` (0-based) rendered on white, scaled so its longer side is
// `max_side` pixels (page rotation applied), as a malloc'd top-down RGB8 buffer
// of (*w_out) x (*h_out) pixels (free with mlxs_pdf_free). NULL when the PDF or
// the page cannot be opened.
unsigned char *mlxs_pdf_render_page(const unsigned char *data, size_t len, int index, int max_side, int *w_out, int *h_out) {
    *w_out = 0;
    *h_out = 0;
    if (max_side < 16) return NULL;
    CGDataProviderRef provider = CGDataProviderCreateWithData(NULL, data, len, NULL);
    if (provider == NULL) return NULL;
    CGPDFDocumentRef doc = CGPDFDocumentCreateWithProvider(provider);
    CGDataProviderRelease(provider);
    if (doc == NULL) return NULL;
    unsigned char *rgb = NULL;
    if (CGPDFDocumentIsEncrypted(doc) && !CGPDFDocumentIsUnlocked(doc) && !CGPDFDocumentUnlockWithPassword(doc, "")) goto done;
    if (index < 0 || (size_t)index >= CGPDFDocumentGetNumberOfPages(doc)) goto done;
    {
        CGPDFPageRef page = CGPDFDocumentGetPage(doc, (size_t)index + 1);
        if (page == NULL) goto done;
        CGRect box = CGPDFPageGetBoxRect(page, kCGPDFCropBox);
        int rot = ((CGPDFPageGetRotationAngle(page) % 360) + 360) % 360;
        double pw = box.size.width, ph = box.size.height;
        if (rot == 90 || rot == 270) {
            double t = pw;
            pw = ph;
            ph = t;
        }
        if (pw < 1 || ph < 1) goto done;
        double scale = (double)max_side / fmax(pw, ph);
        size_t W = (size_t)fmax(1, lround(pw * scale));
        size_t H = (size_t)fmax(1, lround(ph * scale));
        CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
        CGContextRef ctx = CGBitmapContextCreate(NULL, W, H, 8, W * 4, cs, (CGBitmapInfo)kCGImageAlphaNoneSkipLast | kCGBitmapByteOrder32Big);
        CGColorSpaceRelease(cs);
        if (ctx == NULL) goto done;
        CGContextSetRGBFillColor(ctx, 1, 1, 1, 1);
        CGContextFillRect(ctx, CGRectMake(0, 0, W, H));
        CGContextSetInterpolationQuality(ctx, kCGInterpolationHigh);
        CGContextScaleCTM(ctx, (CGFloat)W / pw, (CGFloat)H / ph);
        CGContextConcatCTM(ctx, CGPDFPageGetDrawingTransform(page, kCGPDFCropBox, CGRectMake(0, 0, pw, ph), 0, true));
        CGContextDrawPDFPage(ctx, page);
        const unsigned char *px = CGBitmapContextGetData(ctx);
        rgb = px ? malloc(W * H * 3) : NULL;
        if (rgb != NULL) {
            // Bitmap memory is top row first; drop the padding byte.
            for (size_t i = 0; i < W * H; i++) {
                rgb[i * 3 + 0] = px[i * 4 + 0];
                rgb[i * 3 + 1] = px[i * 4 + 1];
                rgb[i * 3 + 2] = px[i * 4 + 2];
            }
            *w_out = (int)W;
            *h_out = (int)H;
        }
        CGContextRelease(ctx);
    }
done:
    CGPDFDocumentRelease(doc);
    return rgb;
}

void mlxs_pdf_free(void *p) { free(p); }
