// qpprobe — does libavcodec expose a usable quantisation map for this bitstream?
//
// Instrument for the week-1 go/no-go in rppg_codec_research_plan_v3.md §10.
// C1 (reliability gate) needs at least a per-frame QP scalar.
// C2 (bitstream conditioning) needs a per-block map that actually varies over the face.
//
//   build: make
//   usage: qpprobe <file> [--csv | --summary-csv] [--roi x0,y0,x1,y1] [--max-frames N]
//
// Default mode prints one line per frame plus a verdict.
// --csv writes per-block rows to stdout: frame,pict_type,x,y,w,h,qp,in_roi
// --summary-csv writes one row per frame instead of one per block:
//       frame,pict_type,frame_qp,mean_qp,roi_mean,roi_std,roi_min,roi_max,nb_blocks
//       A full clip is ~2.4 M per-block rows per encode; this is ~1800 rows.
//       Fields are empty when the frame has no side data / no blocks / no ROI blocks.
// --max-frames N stops after N frames (default 300, 0 = whole file).
// --roi marks blocks whose centre falls inside the box, so you can compare
//       mean QP on the face against the background. Coordinates in pixels.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/video_enc_params.h>

#define DEFAULT_MAX_FRAMES 300

static int roi_set = 0, rx0, ry0, rx1, ry1;
static int csv_mode = 0, summary_mode = 0;
static int max_frames = DEFAULT_MAX_FRAMES;

static int in_roi(AVVideoBlockParams *b) {
    if (!roi_set) return 0;
    int cx = b->src_x + b->w / 2, cy = b->src_y + b->h / 2;
    return cx >= rx0 && cx < rx1 && cy >= ry0 && cy < ry1;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: qpprobe <file> [--csv | --summary-csv] [--roi x0,y0,x1,y1] [--max-frames N]\n"); return 2; }
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--csv")) csv_mode = 1;
        else if (!strcmp(argv[i], "--summary-csv")) summary_mode = 1;
        else if (!strcmp(argv[i], "--max-frames") && i + 1 < argc) {
            max_frames = atoi(argv[++i]);
            if (max_frames < 0) { fprintf(stderr, "--max-frames wants N >= 0 (0 = no limit)\n"); return 2; }
        }
        else if (!strcmp(argv[i], "--roi") && i + 1 < argc) {
            if (sscanf(argv[++i], "%d,%d,%d,%d", &rx0, &ry0, &rx1, &ry1) != 4) {
                fprintf(stderr, "--roi wants four integers: x0,y0,x1,y1\n");
                return 2;
            }
            // Two corners, not x,y,w,h. Catch the mix-up here rather than
            // reporting an empty ROI as a missing line further down.
            if (rx1 <= rx0 || ry1 <= ry0) {
                fprintf(stderr,
                    "--roi %d,%d,%d,%d is empty: needs x0,y0,x1,y1 with x1>x0 and y1>y0.\n"
                    "If you meant x,y,w,h use --roi %d,%d,%d,%d\n",
                    rx0, ry0, rx1, ry1, rx0, ry0, rx0 + rx1, ry0 + ry1);
                return 2;
            }
            roi_set = 1;
        }
    }

    AVFormatContext *fmt = NULL;
    if (avformat_open_input(&fmt, argv[1], NULL, NULL) < 0) { fprintf(stderr, "open failed\n"); return 1; }
    avformat_find_stream_info(fmt, NULL);
    int vs = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
    if (vs < 0) { fprintf(stderr, "no video stream\n"); return 1; }

    const AVCodec *dec = avcodec_find_decoder(fmt->streams[vs]->codecpar->codec_id);
    AVCodecContext *ctx = avcodec_alloc_context3(dec);
    avcodec_parameters_to_context(ctx, fmt->streams[vs]->codecpar);
    ctx->export_side_data |= AV_CODEC_EXPORT_DATA_VIDEO_ENC_PARAMS;
    if (avcodec_open2(ctx, dec, NULL) < 0) { fprintf(stderr, "decoder open failed\n"); return 1; }

    // Per-block and summary rows are different schemas on the same stdout.
    if (csv_mode && summary_mode) { fprintf(stderr, "--csv and --summary-csv are exclusive\n"); return 2; }
    if (summary_mode) printf("frame,pict_type,frame_qp,mean_qp,roi_mean,roi_std,roi_min,roi_max,nb_blocks\n");
    else if (csv_mode) printf("frame,pict_type,x,y,w,h,qp,in_roi\n");

    AVPacket *pkt = av_packet_alloc();
    AVFrame *frm = av_frame_alloc();
    int frames = 0, frames_with_sd = 0, frames_with_blocks = 0;
    int max_spread = 0; long long spread_sum = 0;
    double roi_delta_sum = 0; int roi_frames = 0;
    // What a cropped C2 model actually sees: the absolute level, and the
    // variation inside the face. ROI-minus-background is a constant to it.
    double qp_mean_sum = 0, roi_mean_sum = 0;
    long long roi_spread_sum = 0; int roi_stat_frames = 0;

    while (av_read_frame(fmt, pkt) >= 0 && (!max_frames || frames < max_frames)) {
        if (pkt->stream_index != vs) { av_packet_unref(pkt); continue; }
        if (avcodec_send_packet(ctx, pkt) == 0) {
            while (avcodec_receive_frame(ctx, frm) == 0 && (!max_frames || frames < max_frames)) {
                char pt = av_get_picture_type_char(frm->pict_type);
                AVFrameSideData *sd = av_frame_get_side_data(frm, AV_FRAME_DATA_VIDEO_ENC_PARAMS);
                if (!sd) {
                    if (summary_mode) printf("%d,%c,,,,,,,\n", frames, pt);
                    if (!csv_mode && !summary_mode) printf("  frame %3d (%c): NO side data\n", frames, pt);
                } else {
                    frames_with_sd++;
                    AVVideoEncParams *p = (AVVideoEncParams *)sd->data;
                    if (!p->nb_blocks) {
                        if (summary_mode) printf("%d,%c,%d,,,,,,0\n", frames, pt, p->qp);
                        if (!csv_mode && !summary_mode) printf("  frame %3d (%c): frame_qp=%d  nb_blocks=0 (scalar only)\n",
                                              frames, pt, p->qp);
                    } else {
                        frames_with_blocks++;
                        int mn = INT_MAX, mx = INT_MIN;
                        int roi_mn = INT_MAX, roi_mx = INT_MIN;
                        long sum = 0, roi_sum = 0, bg_sum = 0;
                        double roi_sq = 0;
                        int roi_n = 0, bg_n = 0;
                        for (unsigned i = 0; i < p->nb_blocks; i++) {
                            AVVideoBlockParams *b = av_video_enc_params_block(p, i);
                            int q = p->qp + b->delta_qp;
                            if (q < mn) mn = q;
                            if (q > mx) mx = q;
                            sum += q;
                            int r = in_roi(b);
                            if (roi_set) {
                                if (r) {
                                    roi_sum += q; roi_sq += (double)q * q; roi_n++;
                                    if (q < roi_mn) roi_mn = q;
                                    if (q > roi_mx) roi_mx = q;
                                } else { bg_sum += q; bg_n++; }
                            }
                            if (csv_mode)
                                printf("%d,%c,%d,%d,%d,%d,%d,%d\n", frames, pt, b->src_x, b->src_y, b->w, b->h, q, r);
                        }
                        int spread = mx - mn;
                        if (spread > max_spread) max_spread = spread;
                        spread_sum += spread;
                        qp_mean_sum += (double)sum / p->nb_blocks;

                        if (summary_mode) {
                            printf("%d,%c,%d,%.4f,", frames, pt, p->qp, (double)sum / p->nb_blocks);
                            if (roi_set && roi_n) {
                                double m = (double)roi_sum / roi_n;
                                double v = roi_sq / roi_n - m * m;
                                printf("%.4f,%.4f,%d,%d", m, v > 0 ? sqrt(v) : 0.0, roi_mn, roi_mx);
                            } else printf(",,,");
                            printf(",%u\n", (unsigned)p->nb_blocks);
                        }

                        double d = 0; int have_delta = 0;
                        if (roi_set && roi_n) {
                            roi_mean_sum += (double)roi_sum / roi_n;
                            roi_spread_sum += roi_mx - roi_mn;
                            roi_stat_frames++;
                            if (bg_n) {
                                d = (double)roi_sum / roi_n - (double)bg_sum / bg_n;
                                roi_delta_sum += d; roi_frames++; have_delta = 1;
                            }
                        }

                        if (!csv_mode && !summary_mode) {
                            printf("  frame %3d (%c): frame_qp=%d  nb_blocks=%u  qp min=%d max=%d mean=%.1f spread=%d",
                                   frames, pt, p->qp, (unsigned)p->nb_blocks, mn, mx, (double)sum / p->nb_blocks, spread);
                            if (have_delta) printf("  roi-bg=%+.1f", d);
                            printf("\n");
                        }
                    }
                }
                frames++;
            }
        }
        av_packet_unref(pkt);
    }

    if (!csv_mode && !summary_mode) {
        printf("\n  --- %s\n", argv[1]);
        printf("  frames decoded          : %d\n", frames);
        printf("  with QP side data       : %d\n", frames_with_sd);
        printf("  with a per-block map    : %d\n", frames_with_blocks);
        if (frames_with_blocks) {
            printf("  mean QP (whole frame)   : %.1f\n", qp_mean_sum / frames_with_blocks);
            printf("  mean within-frame spread: %.1f QP steps (max %d)\n",
                   (double)spread_sum / frames_with_blocks, max_spread);
            // The two lines a cropped C2 model is actually conditioned on.
            if (roi_stat_frames) {
                printf("  mean QP inside ROI      : %.1f\n", roi_mean_sum / roi_stat_frames);
                printf("  mean within-ROI spread  : %.1f QP steps\n",
                       (double)roi_spread_sum / roi_stat_frames);
            }
            if (roi_frames)
                printf("  mean ROI-background QP  : %+.1f steps  (roi %d,%d -> %d,%d)\n",
                       roi_delta_sum / roi_frames, rx0, ry0, rx1, ry1);
            else if (roi_set)
                printf("  mean ROI-background QP  : NOT MEASURED - no blocks fell inside "
                       "%d,%d -> %d,%d; is the box off-frame?\n", rx0, ry0, rx1, ry1);
            else
                printf("  mean ROI-background QP  : no --roi given, so the face question is unanswered\n");
        }
        // Gate A: a spatial map exists at all.  Gate C: at least a scalar.
        const char *verdict = frames_with_blocks ? (max_spread > 0 ? "PASS  - per-block map, spatially varying (C1 + C2 viable)"
                                                                  : "WEAK  - per-block map but flat (C1 ok, C2 uninformative)")
                            : frames_with_sd     ?  "PARTIAL - frame-level QP scalar only (C1 only, C2 dead)"
                                                 :  "FAIL  - no QP exposed by this decoder";
        printf("  VERDICT                 : %s\n", verdict);
    }

    av_frame_free(&frm);
    av_packet_free(&pkt);
    avcodec_free_context(&ctx);
    avformat_close_input(&fmt);
    return 0;
}
