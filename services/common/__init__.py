# Package Common - Modules partagés entre services
"""
Ce package centralise les configurations et connexions partagées
pour éviter la duplication de code entre les services.
"""

from .config import (
    SAMPLE_RATE,
    CHUNK_DURATION,
    DECIMATE_FACTOR,
    AMPLITUDE_SCALE,
    IMG_SIZE,
    BUBBLE_PARAMS,
    CLOGGING_LEVELS,
    LABEL_TO_INDEX,
    INDEX_TO_LABEL,
    LABEL_NOISE_RATE,
    TimescaleConfig,
    MongoConfig,
    MinIOConfig,
)

from .db_connections import (
    get_timescale_connection,
    get_mongo_client,
    get_mongo_collection,
    get_minio_client,
)

from .signal_processing import (
    generate_signal,
    insert_batch,
)
