// set for [2] second window, NOT 1 second. change by searching for "2SECOND"
// redefine ARM_COMMAND to "" to perform gesture recongition on pressing Enter
// Input selected arming command and press Enter. A 2-second window will begin to perform a gesture. Predicted gesture will be printed.

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stddef.h>

#include "esp_log.h"
#include "driver/i2c.h"
#include "esp_timer.h"
#include "driver/uart.h"

static const char *TAG = "gesture_capture";

#define I2C_MASTER_SCL_IO           1
#define I2C_MASTER_SDA_IO           0
#define I2C_MASTER_NUM              0
#define I2C_MASTER_FREQ_HZ          400000
#define I2C_MASTER_TX_BUF_DISABLE   0
#define I2C_MASTER_RX_BUF_DISABLE   0
#define I2C_MASTER_TIMEOUT_MS       1000

#define MPU6050_SENSOR_ADDR         0x68
#define MPU6050_WHO_AM_I_REG_ADDR   0x75
#define MPU6050_PWR_MGMT_1_REG_ADDR 0x6B
#define MPU6050_RESET_BIT           7
#define MPU6050_GYRO_CONFIG         0x1B
#define MPU6050_ACCEL_CONFIG        0x1C
#define MPU6050_MEASURE_REG_ADDR    0x3B

#define GRAV_ACCEL 9.80665f
#define ACCEL_SEN  16384.0f
#define GYRO_SEN   65.5f

#define GX_BIAS -3.934806f
#define GY_BIAS  0.417461f
#define GZ_BIAS  0.531037f

/* 2SECOND window*/
#define SAMPLE_RATE_HZ      250
#define CAPTURE_SECONDS     2
#define NUM_SAMPLES         (SAMPLE_RATE_HZ * CAPTURE_SECONDS)
#define SAMPLE_PERIOD_US    (1000000 / SAMPLE_RATE_HZ)


/* 1SECOND window
#define SAMPLE_RATE_HZ      250
#define CAPTURE_SECONDS     1
#define NUM_SAMPLES         (SAMPLE_RATE_HZ * CAPTURE_SECONDS)
#define SAMPLE_PERIOD_US    (1000000 / SAMPLE_RATE_HZ)
 */

#define ARM_COMMAND "ARM" // select command to input for gesture recognition, use "" for Enter key only
#define CSV_COMMAND "CSV"

#define CMD_UART UART_NUM_0
#define CMD_BUF_LEN 64

typedef struct {
    int16_t ax_raw;
    int16_t ay_raw;
    int16_t az_raw;
    int16_t gx_raw;
    int16_t gy_raw;
    int16_t gz_raw;
} imu_sample_t;

typedef struct {
    float ax;
    float ay;
    float az;
    float gx;
    float gy;
    float gz;
} imu_sample_f32_t;

const char *predict_gesture_or_unknown(const imu_sample_f32_t *samples, size_t n);

static imu_sample_t samples[NUM_SAMPLES];
static imu_sample_f32_t infer_samples[NUM_SAMPLES];

static esp_err_t mpu6050_register_read(uint8_t reg_addr, uint8_t *measurements, size_t len)
{
    return i2c_master_write_read_device(I2C_MASTER_NUM, MPU6050_SENSOR_ADDR, &reg_addr, 1, measurements, len,
                                        I2C_MASTER_TIMEOUT_MS / portTICK_PERIOD_MS);
}

static esp_err_t mpu6050_register_write_byte(uint8_t reg_addr, uint8_t value)
{
    uint8_t write_buf[2] = {reg_addr, value};
    return i2c_master_write_to_device(I2C_MASTER_NUM, MPU6050_SENSOR_ADDR, write_buf, sizeof(write_buf),
                                      I2C_MASTER_TIMEOUT_MS / portTICK_PERIOD_MS);
}

static esp_err_t i2c_master_init(void)
{
    i2c_config_t conf = {
        .mode = I2C_MODE_MASTER,
        .sda_io_num = I2C_MASTER_SDA_IO,
        .scl_io_num = I2C_MASTER_SCL_IO,
        .sda_pullup_en = GPIO_PULLUP_ENABLE,
        .scl_pullup_en = GPIO_PULLUP_ENABLE,
        .master.clk_speed = I2C_MASTER_FREQ_HZ,
    };

    i2c_param_config(I2C_MASTER_NUM, &conf);
    return i2c_driver_install(I2C_MASTER_NUM, conf.mode, I2C_MASTER_RX_BUF_DISABLE, I2C_MASTER_TX_BUF_DISABLE, 0);
}

static void init_imu(void)
{
    uint8_t who = 0;
    ESP_ERROR_CHECK(i2c_master_init());
    ESP_ERROR_CHECK(mpu6050_register_read(MPU6050_WHO_AM_I_REG_ADDR, &who, 1));
    ESP_LOGI(TAG, "WHO_AM_I = 0x%02X", who);

    ESP_ERROR_CHECK(mpu6050_register_write_byte(MPU6050_PWR_MGMT_1_REG_ADDR, 1 << MPU6050_RESET_BIT));
    vTaskDelay(pdMS_TO_TICKS(100));
    ESP_ERROR_CHECK(mpu6050_register_write_byte(MPU6050_PWR_MGMT_1_REG_ADDR, 0x00));
    vTaskDelay(pdMS_TO_TICKS(100));
    ESP_ERROR_CHECK(mpu6050_register_write_byte(MPU6050_ACCEL_CONFIG, 0x00));
    ESP_ERROR_CHECK(mpu6050_register_write_byte(MPU6050_GYRO_CONFIG, 0x08));
}

static void capture_window(void)
{
    uint8_t measurements[14];
    int64_t t_next = esp_timer_get_time();

    for (int i = 0; i < NUM_SAMPLES; i++) {
        while (esp_timer_get_time() < t_next) {
        }

        ESP_ERROR_CHECK(mpu6050_register_read(MPU6050_MEASURE_REG_ADDR, measurements, 14));

        samples[i].ax_raw = ((int16_t)measurements[0] << 8) | measurements[1];
        samples[i].ay_raw = ((int16_t)measurements[2] << 8) | measurements[3];
        samples[i].az_raw = ((int16_t)measurements[4] << 8) | measurements[5];
        samples[i].gx_raw = ((int16_t)measurements[8] << 8) | measurements[9];
        samples[i].gy_raw = ((int16_t)measurements[10] << 8) | measurements[11];
        samples[i].gz_raw = ((int16_t)measurements[12] << 8) | measurements[13];

        t_next += SAMPLE_PERIOD_US;
    }
}

static void prepare_inference_window(void)
{
    for (int i = 0; i < NUM_SAMPLES; i++) {
        infer_samples[i].ax = (samples[i].ax_raw * GRAV_ACCEL) / ACCEL_SEN;
        infer_samples[i].ay = (samples[i].ay_raw * GRAV_ACCEL) / ACCEL_SEN;
        infer_samples[i].az = (samples[i].az_raw * GRAV_ACCEL) / ACCEL_SEN;
        infer_samples[i].gx = (samples[i].gx_raw / GYRO_SEN) - GX_BIAS;
        infer_samples[i].gy = (samples[i].gy_raw / GYRO_SEN) - GY_BIAS;
        infer_samples[i].gz = (samples[i].gz_raw / GYRO_SEN) - GZ_BIAS;
    }
}

static void print_window_csv(void)
{
    printf("time_us,ax,ay,az,gx,gy,gz\n");
    for (int i = 0; i < NUM_SAMPLES; i++) {
        printf("%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
               i * SAMPLE_PERIOD_US,
               infer_samples[i].ax,
               infer_samples[i].ay,
               infer_samples[i].az,
               infer_samples[i].gx,
               infer_samples[i].gy,
               infer_samples[i].gz);
    }
}

static void init_cmd_uart(void)
{
    const uart_config_t uart_config = {
        .baud_rate = 115200,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT,
    };

    ESP_ERROR_CHECK(uart_driver_install(CMD_UART, 1024, 0, 0, NULL, 0));
    ESP_ERROR_CHECK(uart_param_config(CMD_UART, &uart_config));
}

void app_main(void)
{
    uint8_t ch;
    char cmd[CMD_BUF_LEN];
    int idx = 0;

    init_imu();
    init_cmd_uart();

    ESP_LOGI(TAG, "Ready. Send '%s' to classify a 2-second window, or '%s' to dump CSV.", ARM_COMMAND, CSV_COMMAND);

    while (1) {
        int n = uart_read_bytes(CMD_UART, &ch, 1, pdMS_TO_TICKS(20));
        if (n <= 0) {
            continue;
        }

        if (ch == '\r' || ch == '\n') {
            if (idx == 0) {
                continue;
            }

            cmd[idx] = '\0';
            idx = 0;

            if (strcmp(cmd, ARM_COMMAND) == 0) {
                ESP_LOGI(TAG, "Capture armed. Record now.");
                capture_window();
                prepare_inference_window();

                const char *label = predict_gesture_or_unknown(infer_samples, NUM_SAMPLES);
                ESP_LOGI(TAG, "Predicted gesture: %s", label);
                printf("PREDICTED:%s\n", label); // to laptop

                ESP_LOGI(TAG, "Capture complete.");
            } else if (strcmp(cmd, CSV_COMMAND) == 0) {
                ESP_LOGI(TAG, "Capture armed for CSV dump.");
                capture_window();
                prepare_inference_window();
                print_window_csv();
                ESP_LOGI(TAG, "CSV dump complete.");
            } else {
                ESP_LOGI(TAG, "Unknown command: '%s'", cmd);
            }
        } else {
            if (idx < CMD_BUF_LEN - 1) {
                cmd[idx++] = (char)ch;
            } else {
                idx = 0;
            }
        }
    }
}
