// PDF text extraction for mlx-serve's `document` blocks (src/pdf.zig), via PDFKit.
// ARC objc; macOS only (the Linux / iOS builds compile src/pdf.zig's stub instead).
#import <Foundation/Foundation.h>
#import <PDFKit/PDFKit.h>
#include <stdlib.h>
#include <string.h>

// The document's text, one "--- page N ---" section per page, as a malloc'd
// UTF-8 string of *len_out bytes (free with mlxs_pdf_free). NULL when the bytes
// are not a PDF PDFKit can open, or it is encrypted with a non-empty password;
// *pages_out is still set when the page count is known. *text_pages_out counts
// the pages that had any text (0 for a scanned document).
char *mlxs_pdf_text(const unsigned char *data, size_t len, int *pages_out, int *text_pages_out, size_t *len_out) {
    *pages_out = 0;
    *text_pages_out = 0;
    *len_out = 0;
    @autoreleasepool {
        NSData *bytes = [NSData dataWithBytesNoCopy:(void *)data length:len freeWhenDone:NO];
        PDFDocument *doc = [[PDFDocument alloc] initWithData:bytes];
        if (doc == nil) return NULL;
        *pages_out = (int)doc.pageCount;
        if (doc.isLocked && ![doc unlockWithPassword:@""]) return NULL;
        NSMutableString *out = [NSMutableString string];
        NSCharacterSet *ws = [NSCharacterSet whitespaceAndNewlineCharacterSet];
        NSInteger n = doc.pageCount;
        int with_text = 0;
        for (NSInteger i = 0; i < n; i++) {
            PDFPage *page = [doc pageAtIndex:i];
            NSString *s = [page.string stringByTrimmingCharactersInSet:ws] ?: @"";
            if (i > 0) [out appendString:@"\n\n"];
            [out appendFormat:@"--- page %ld ---", (long)(i + 1)];
            if (s.length > 0) {
                with_text++;
                [out appendString:@"\n"];
                [out appendString:s];
            }
        }
        const char *utf8 = out.UTF8String;
        if (utf8 == NULL) return NULL;
        size_t ulen = strlen(utf8);
        char *res = malloc(ulen + 1);
        if (res == NULL) return NULL;
        memcpy(res, utf8, ulen + 1);
        *text_pages_out = with_text;
        *len_out = ulen;
        return res;
    }
}

void mlxs_pdf_free(char *p) { free(p); }
