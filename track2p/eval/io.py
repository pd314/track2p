from pathlib import Path
import numpy as np

from ..logs import get_logger
logger = get_logger(__name__)


def load_alldays_f1_values(base_path, animals, conditions):

    base_path = Path(base_path)

    logger.info("Loading F1 values across all days")

    f1_values = {animal: [] for animal in animals}

    for animal in animals:

        for condition in conditions:

            file_path = base_path / animal / condition / "metrics_t2p_all_days.npy"

            if not file_path.exists():
                logger.warning(f"Missing file: {file_path}")
                continue

            logger.debug(f"Loading: {file_path}")

            metrics = np.load(file_path, allow_pickle=True)

            try:
                f1_value = metrics[np.where(metrics[:, 0] == 'F1')[0][0], 1]
                f1_values[animal].append(f1_value)

                logger.debug(f"Animal={animal} | F1={f1_value}")

            except Exception as e:
                logger.error(f"Failed extracting F1 from {file_path}", exc_info=True)
                raise

    logger.info("Completed F1 loading")

    return f1_values


def load_alldays_ct_values(base_path, animals, conditions, ct_type='CT'):

    base_path = Path(base_path)

    logger.info(f"Loading CT/accuracy values | type={ct_type}")

    ct_values = {animal: [] for animal in animals}
    acc_values = {animal: [] for animal in animals}

    for animal in animals:

        for condition in conditions:

            file_path = base_path / animal / condition / f"result_{ct_type}.npy"

            if not file_path.exists():
                logger.warning(f"Missing file: {file_path}")
                continue

            logger.debug(f"Loading: {file_path}")

            metrics = np.load(file_path, allow_pickle=True)

            logger.debug(f"Raw metrics shape: {np.shape(metrics)}")

            try:
                ct = metrics[0][1:]
                acc = metrics[1][1:]

                ct_values[animal].append(np.array(ct))
                acc_values[animal].append(np.array(acc))

            except Exception as e:
                logger.error(f"Failed parsing CT/ACC from {file_path}", exc_info=True)
                raise

    logger.info("Completed CT/accuracy loading")

    return ct_values, acc_values


def load_pairwise_f1_values(base_path, animals, condition):

    base_path = Path(base_path)

    logger.info(f"Loading pairwise F1 values | condition={condition}")

    f1_values = {animal: [] for animal in animals}

    for animal in animals:

        if condition == 'pw_reg':
            file_path = base_path / animal / "metrics_table_pw_registration.npy"
        else:
            file_path = base_path / animal / condition / "metrics_table_pairs.npy"

        if not file_path.exists():
            logger.warning(f"Missing file: {file_path}")
            continue

        logger.debug(f"Loading: {file_path}")

        metrics = np.load(file_path, allow_pickle=True)

        try:
            f1_scores = metrics[7, 1:].astype(float)
            f1_values[animal] = f1_scores

            logger.debug(f"Animal={animal} | F1 shape={f1_scores.shape}")

        except Exception as e:
            logger.error(f"Failed parsing pairwise F1 from {file_path}", exc_info=True)
            raise

    logger.info("Completed pairwise F1 loading")

    return f1_values