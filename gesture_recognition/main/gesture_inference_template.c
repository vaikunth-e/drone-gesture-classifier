// set for [2] second window, NOT 1 second. change by searching for "2SECOND"

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "gesture_model_params.h" // 2SECOND
//#include "1sec_gesture_model_params.h"

#define IMU_CHANNELS 6
#define FEATURES_PER_CHANNEL 10
#define UNKNOWN_SCORE_THRESHOLD 0.40f
#define UNKNOWN_MARGIN_THRESHOLD 0.60f

typedef struct {
    float ax;
    float ay;
    float az;
    float gx;
    float gy;
    float gz;
} imu_sample_f32_t;

static float get_channel_value(const imu_sample_f32_t *samples, size_t idx, size_t channel)
{
    switch (channel) {
        case 0: return samples[idx].ax;
        case 1: return samples[idx].ay;
        case 2: return samples[idx].az;
        case 3: return samples[idx].gx;
        case 4: return samples[idx].gy;
        default: return samples[idx].gz;
    }
}

static int count_zero_crossings_centered(const imu_sample_f32_t *samples, size_t n, size_t channel, float mean)
{
    int count = 0;
    int have_prev = 0;
    int prev_sign = 0;

    for (size_t i = 0; i < n; i++) {
        float centered = get_channel_value(samples, i, channel) - mean;
        int sign = (centered > 0.0f) - (centered < 0.0f);
        if (sign == 0) {
            continue;
        }
        if (have_prev && sign != prev_sign) {
            count++;
        }
        prev_sign = sign;
        have_prev = 1;
    }
    return count;
}

static void compute_features(const imu_sample_f32_t *samples, size_t n, float out[GESTURE_NUM_FEATURES])
{
    size_t f = 0;

    for (size_t c = 0; c < IMU_CHANNELS; c++) {
        float sum = 0.0f;
        float sum_sq = 0.0f;
        float min_v = FLT_MAX;
        float max_v = -FLT_MAX;
        float peak_abs = 0.0f;
        float sum_abs_diff = 0.0f;
        float prev = 0.0f;
        int have_prev = 0;

        for (size_t i = 0; i < n; i++) {
            float x = get_channel_value(samples, i, c);
            sum += x;
            sum_sq += x * x;
            if (x < min_v) min_v = x;
            if (x > max_v) max_v = x;
            if (fabsf(x) > peak_abs) peak_abs = fabsf(x);
            if (have_prev) {
                sum_abs_diff += fabsf(x - prev);
            }
            prev = x;
            have_prev = 1;
        }

        float mean = sum / (float)n;
        float energy = sum_sq / (float)n;
        float var = energy - mean * mean;
        if (var < 0.0f) var = 0.0f;
        float std = sqrtf(var);
        float range = max_v - min_v;
        float rms = sqrtf(energy);
        float mean_abs_diff = (n > 1) ? sum_abs_diff / (float)(n - 1) : 0.0f;
        float zero_crossings_centered = (float)count_zero_crossings_centered(samples, n, c, mean);

        out[f++] = mean;
        out[f++] = std;
        out[f++] = min_v;
        out[f++] = max_v;
        out[f++] = range;
        out[f++] = energy;
        out[f++] = peak_abs;
        out[f++] = rms;
        out[f++] = zero_crossings_centered;
        out[f++] = mean_abs_diff;
    }
}

static void scale_features(float x[GESTURE_NUM_FEATURES])
{
    for (size_t i = 0; i < GESTURE_NUM_FEATURES; i++) {
        x[i] = (x[i] - GESTURE_FEATURE_MEAN[i]) / GESTURE_FEATURE_SCALE[i];
    }
}

static void compute_scores(const float x[GESTURE_NUM_FEATURES], float scores[GESTURE_NUM_CLASSES])
{
    for (size_t c = 0; c < GESTURE_NUM_CLASSES; c++) {
        float score = GESTURE_BIAS[c];
        for (size_t i = 0; i < GESTURE_NUM_FEATURES; i++) {
            score += GESTURE_WEIGHT[c][i] * x[i];
        }
        scores[c] = score;
    }
}

const char *predict_gesture_or_unknown(const imu_sample_f32_t *samples, size_t n)
{
    float features[GESTURE_NUM_FEATURES];
    float scores[GESTURE_NUM_CLASSES];

    int best_idx = -1;
    int second_idx = -1;
    float best_score = -FLT_MAX;
    float second_score = -FLT_MAX;

    compute_features(samples, n, features);
    scale_features(features);
    compute_scores(features, scores);

    for (size_t c = 0; c < GESTURE_NUM_CLASSES; c++) {
        float s = scores[c];
        if (s > best_score) {
            second_score = best_score;
            second_idx = best_idx;
            best_score = s;
            best_idx = (int)c;
        } else if (s > second_score) {
            second_score = s;
            second_idx = (int)c;
        }
    }

    if (best_idx < 0 || second_idx < 0) {
        return "unknown";
    }

    float margin = best_score - second_score;
    if (best_score < UNKNOWN_SCORE_THRESHOLD || margin < UNKNOWN_MARGIN_THRESHOLD) {
        return "unknown";
    }

    return GESTURE_LABELS[best_idx];
}
