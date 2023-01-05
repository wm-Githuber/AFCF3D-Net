import numpy as np


def initialize_metrics():
    """Generates a dictionary of metrics with metrics as keys
       and empty lists as values

    Returns
    -------
    dict
        a dictionary of metrics

    """
    metrics = {
        'cd_losses': [],
        # 'cd_corrects': [],
        'cd_precisions': [],
        'cd_recalls': [],
        'cd_f1scores': [],
        # 'learning_rate': [],
    }

    return metrics

def get_mean_metrics(metric_dict):
    return {k: np.mean(v) for k, v in metric_dict.items()}

def set_metrics(metric_dict, cd_loss, cd_report):
    metric_dict['cd_losses'].append(cd_loss.item())
    # metric_dict['cd_corrects'].append(cd_corrects.item())
    metric_dict['cd_precisions'].append(cd_report[0])
    metric_dict['cd_recalls'].append(cd_report[1])
    metric_dict['cd_f1scores'].append(cd_report[2])
    # metric_dict['learning_rate'].append(lr)

    return metric_dict
