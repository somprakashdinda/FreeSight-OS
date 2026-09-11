/**
 * neuromorphic_dvs_engine.cpp
 * ============================
 * Asynchronous Neuromorphic Event-Camera HAL & Saccade Motion Filter.
 * Processes high-frequency microsecond pixel polarity events (>1000 Hz)
 * from Dynamic Vision Sensors (DVS / Event Cameras).
 *
 * Part of the v6.0 Neuromorphic & Cross-Platform Enterprise Engine.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(_WIN32)
#define DVS_EXPORT __declspec(dllexport)
#else
#define DVS_EXPORT __attribute__((visibility("default")))
#endif

#pragma pack(push, 1)
struct DVSEvent {
    uint16_t x;
    uint16_t y;
    uint64_t timestamp_us;
    int8_t polarity; // +1 = brighter, -1 = darker
};
#pragma pack(pop)

class NeuromorphicDVSEngine {
public:
    NeuromorphicDVSEngine()
        : last_event_timestamp_us_(0),
          accumulated_events_(0),
          last_centroid_x_(0.0f),
          last_centroid_y_(0.0f) {}

    void ProcessEventPacket(const DVSEvent* events, size_t count, float* out_velocity_px_sec) {
        if (!events || count == 0) {
            if (out_velocity_px_sec) *out_velocity_px_sec = 0.0f;
            return;
        }

        double sum_x = 0.0;
        double sum_y = 0.0;
        uint64_t t_min = events[0].timestamp_us;
        uint64_t t_max = events[count - 1].timestamp_us;

        for (size_t i = 0; i < count; ++i) {
            sum_x += events[i].x;
            sum_y += events[i].y;
        }

        float current_centroid_x = static_cast<float>(sum_x / count);
        float current_centroid_y = static_cast<float>(sum_y / count);

        float velocity = 0.0f;
        if (accumulated_events_ > 0 && t_max > last_event_timestamp_us_) {
            float dt_sec = static_cast<float>(t_max - last_event_timestamp_us_) / 1000000.0f;
            if (dt_sec > 0.000001f) {
                float dx = current_centroid_x - last_centroid_x_;
                float dy = current_centroid_y - last_centroid_y_;
                velocity = std::sqrt(dx * dx + dy * dy) / dt_sec;
            }
        }

        last_centroid_x_ = current_centroid_x;
        last_centroid_y_ = current_centroid_y;
        last_event_timestamp_us_ = t_max;
        accumulated_events_ += count;

        if (out_velocity_px_sec) {
            *out_velocity_px_sec = velocity;
        }
    }

    uint64_t GetAccumulatedEvents() const {
        return accumulated_events_;
    }

private:
    uint64_t last_event_timestamp_us_;
    uint64_t accumulated_events_;
    float last_centroid_x_;
    float last_centroid_y_;
};

extern "C" {
    DVS_EXPORT void* DVS_CreateEngine() {
        return new NeuromorphicDVSEngine();
    }

    DVS_EXPORT void DVS_ProcessEvents(void* engine, const DVSEvent* events, size_t count, float* out_vel) {
        if (!engine) return;
        auto* ptr = static_cast<NeuromorphicDVSEngine*>(engine);
        ptr->ProcessEventPacket(events, count, out_vel);
    }

    DVS_EXPORT void DVS_DestroyEngine(void* engine) {
        if (!engine) return;
        delete static_cast<NeuromorphicDVSEngine*>(engine);
    }
}
