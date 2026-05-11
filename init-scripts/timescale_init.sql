-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create table for raw audio data
-- Added 'label' to store the ground truth clogging level (0, 20, 40, 60, 80)
CREATE TABLE audio_data (
    time        TIMESTAMPTZ       NOT NULL,
    amplitude   REAL              NOT NULL,
    label       INTEGER           NOT NULL
);

-- Turn it into a hypertable
SELECT create_hypertable('audio_data', 'time');

-- Create retention policy (e.g., 24 hours as requested)
SELECT add_retention_policy('audio_data', INTERVAL '24 hours');
