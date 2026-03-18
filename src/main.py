from experiments.bank import main as bank_main
from experiments.face_temp import main as face_temp_main
from experiments.parkinsons import main as parkinsons_main
from experiments.thyroid_cancer import main as thyroid_cancer_main
from experiments.wine import main as wine_main
from experiments.rescue import main as rescue_main
from experiments.rescue_classification import main as rescue_classification_main
from experiments.piano import main as piano_main


def main():
    # print("\n" + "=" * 80)
    # print("RUNNING BANK MARKETING EXPERIMENTS")
    # print("=" * 80)
    # bank_main()

    # # print("\n" + "=" * 80)
    # # print("RUNNING FACE TEMPERATURE EXPERIMENTS")
    # # print("=" * 80)
    # # face_temp_main()

    # # print("\n" + "=" * 80)
    # # print("RUNNING PARKINSON'S TELEMONITORING EXPERIMENTS")
    # # print("=" * 80)
    # # parkinsons_main()

    # print("\n" + "=" * 80)
    # print("RUNNING THYROID CANCER EXPERIMENTS")
    # print("=" * 80)
    # thyroid_cancer_main()

    # print("\n" + "=" * 80)
    # print("RUNNING WINE QUALITY EXPERIMENTS")
    # print("=" * 80)
    # wine_main()

    # print("\n" + "=" * 80)
    # print("RUNNING RESCUE EXPERIMENT (REGRESSION)")
    # print("=" * 80)
    # rescue_main()

    print("\n" + "=" * 80)
    print("RUNNING RESCUE CLASSIFICATION EXPERIMENT")
    print("=" * 80)
    rescue_classification_main()

    # print("\n" + "=" * 80)
    # print("RUNNING PIANO EMOTION CLASSIFICATION EXPERIMENT")
    # print("=" * 80)
    # piano_main()


if __name__ == "__main__":
    main()