from datasets import load_dataset, load_from_disk

import config


def main():
    regulations = load_from_disk(config.HF_KBs_path)
    cases = load_from_disk(config.HF_cases_path)

    hipaa_kb = regulations["HIPAA"]
    hipaa_cases = cases["HIPAA"]

    # convert all cases content into embedding vectors. Each case correspond to one sentence vector
    for case in hipaa_cases:
        case_content = case["case_content"]

        # split case content into sentences
        pass


if __name__ == "__main__":
    main()
