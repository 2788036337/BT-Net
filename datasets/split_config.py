"""Dataset paths and the scene split used by the BT-Net experiments."""

TRAIN_DATASET = "datasets/train_dataset.h5"
VAL_DATASET = "datasets/val_dataset.h5"
TEST_DATASET = "datasets/test_dataset.h5"

TRAIN_SCENES = [
    "glass_tiles_ms",
    "fake_and_real_strawberries_ms",
    "fake_and_real_food_ms",
    "real_and_fake_apples_ms",
    "oil_painting_ms",
    "demo_cave_patches",
    "cloth_ms",
    "fake_and_real_lemons_ms",
    "fake_and_real_lemon_slices_ms",
    "fake_and_real_sushi_ms",
    "fake_and_real_beers_ms",
    "pompoms_ms",
    "stuffed_toys_ms",
    "jelly_beans_ms",
    "fake_and_real_tomatoes_ms",
    "real_and_fake_peppers_ms",
    "beads_ms",
    "fake_and_real_peppers_ms",
    "flowers_ms",
    "cd_ms",
]

VAL_SCENES = [
    "feathers_ms",
    "sponges_ms",
    "clay_ms",
    "photo_and_face_ms",
    "egyptian_statue_ms",
]

TEST_SCENES = [
    "face_ms",
    "paints_ms",
    "balloons_ms",
    "chart_and_stuffed_toy_ms",
    "hairs_ms",
]

TRAIN_RATIO = 20
VAL_RATIO = 5
TEST_RATIO = 5
TOTAL_SAMPLES = 30
DATA_SOURCES = ["GT", "HRMSI", "LRHSI"]


if __name__ == "__main__":
    print(f"Training scenes: {len(TRAIN_SCENES)}")
    print(f"Validation scenes: {len(VAL_SCENES)}")
    print(f"Test scenes: {len(TEST_SCENES)}")
    print(f"Total scenes: {TOTAL_SAMPLES}")
