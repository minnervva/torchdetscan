import torch
import pathlib
import pandas as pd
import time
import numpy as np
from scipy import stats


cpu = torch.device("cpu")


def nn_latency(model, input, iterations):
    torch.use_deterministic_algorithms(mode=False)
    nd_latency = np.array([])
    for _ in range(iterations):
        start_nd = time.time()
        model(input)
        end_nd = time.time()
        nd_latency = np.append(nd_latency, end_nd - start_nd)
    try:
        torch.use_deterministic_algorithms(mode=True)
        det_latency = np.array([])
        for _ in range(iterations):
            start_det = time.time()
            model(input)
            end_det = time.time()
            det_latency = np.append(det_latency, end_det - start_det)
    except:
        return {
            "det_latency_mean": None,
            "det_latency_std": None,
            "nd_latency_mean": np.mean(nd_latency),
            "nd_latency_std": np.std(nd_latency),
            "latency_tstat": None,
            "latency_pvalue": None,
        }

    tstat, pvalue = stats.ttest_ind(det_latency, nd_latency)

    return {
        "det_latency_mean": np.mean(det_latency),
        "det_latency_std": np.std(det_latency),
        "nd_latency_mean": np.mean(nd_latency),
        "nd_latency_std": np.std(nd_latency),
        "latency_tstat": tstat,
        "latency_pvalue": pvalue,
    }


def nn_get_data_backward(model, input, iterations):
    try:
        torch.use_deterministic_algorithms(mode=True)
        input.requires_grad = True
        output = model(input)
        grad = torch.ones(output.shape)
        output.backward(grad)
        base_output = input.grad.to(cpu)
    except:
        base_output = None
    torch.use_deterministic_algorithms(mode=False)
    outputs = []
    for _ in range(iterations):
        input.requires_grad = True
        output = model(input)
        grad = torch.ones(output.shape)
        output.backward(grad)
        outputs.append(input.grad.to(cpu))
    return base_output, outputs


def nn_get_data_forward(model, input, iterations):
    try:
        torch.use_deterministic_algorithms(mode=True)
        base_output = model(input).to(cpu)
    except:
        base_output = None
    torch.use_deterministic_algorithms(mode=False)
    outputs = []
    for _ in range(iterations):
        outputs.append(model(input).to(cpu))
    return base_output, outputs


def nn_get_data(model, input, iterations, backward):
    if not backward:
        return nn_get_data_forward(model, input, iterations)
    else:
        return nn_get_data_backward(model, input, iterations)


def func_latency(func, input, iterations):
    torch.use_deterministic_algorithms(mode=False)
    nd_latency = np.array([])
    for _ in range(iterations):
        start_nd = time.time()
        func(**input)
        end_nd = time.time()
        nd_latency = np.append(nd_latency, end_nd - start_nd)
    try:
        torch.use_deterministic_algorithms(mode=True)
        det_latency = np.array([])
        for _ in range(iterations):
            start_det = time.time()
            func(**input)
            end_det = time.time()
            det_latency = np.append(det_latency, end_det - start_det)
        tstat, pvalue = stats.ttest_ind(det_latency, nd_latency)
        return {
            "det_latency_mean": np.mean(det_latency),
            "det_latency_std": np.std(det_latency),
            "nd_latency_mean": np.mean(nd_latency),
            "nd_latency_std": np.std(nd_latency),
            "latency_tstat": tstat,
            "latency_pvalue": pvalue,
        }
    except:
        return {
            "det_latency_mean": None,
            "det_latency_std": None,
            "nd_latency_mean": np.mean(nd_latency),
            "nd_latency_std": np.std(nd_latency),
            "latency_tstat": None,
            "latency_pvalue": None,
        }


def func_get_data(func, input, iterations):
    try:
        torch.use_deterministic_algorithms(mode=True)
        base_output = func(**input).to(cpu)
    except:
        base_output = None

    torch.use_deterministic_algorithms(mode=False)
    outputs = []
    for _ in range(iterations):
        outputs.append(func(**input).to(cpu))
    return base_output, outputs


def compute_matrix_norms(outputs):
    nuc_norms = []
    fro_norms = []
    for output in outputs:
        nuc_norms.append(torch.norm(output, "nuc"))
        fro_norms.append(torch.norm(output, "fro"))
    return nuc_norms, fro_norms


def num_different_elems(a, b):
    zero = torch.tensor([0.0]).to(torch.float32)[0]
    return torch.sum(((a - b) != zero).to(torch.int)).item()


def sanjif_error(a, b):
    epsilon = 1e-8
    relative_difference = torch.abs(
        (a - b) / (a + epsilon)
    )  # use epsilon to prevent dicision by zero
    # std_dev_relative_difference = torch.std(input=relative_difference, correction=False, keepdim=False)
    average_relative_difference = torch.mean(relative_difference)

    return average_relative_difference.item()


def err_metrics():
    return [
        "det_norm",
        "norm_mean",
        "norm_std",
        "num_diff_mean",
        "num_diff_std",
        "errs_mean",
        "errs_std",
        "norm_tstat",
        "norm_pvalue",
    ]


def all_error_metrics(base_output, outputs):
    norms = np.array([torch.norm(tensor).item() for tensor in outputs])
    num_dif = np.array(
        [num_different_elems(outputs[0], output) for output in outputs[1:]]
    )
    errs = np.array([sanjif_error(outputs[0], output) for output in outputs[1:]])

    if base_output == None:
        return {
            "det_norm": None,
            "norm_mean": np.mean(norms),
            "norm_std": np.std(norms),
            "num_diff_mean": np.mean(num_dif),
            "num_diff_std": np.std(num_dif),
            "errs_mean": np.mean(errs),
            "errs_std": np.std(errs),
            "norm_tstat": None,
            "norm_pvalue": None,
        }
    norm_tstat, norm_pvalue = stats.ttest_1samp(
        norms, popmean=torch.norm(base_output).item()
    )
    return {
        "det_norm": torch.norm(base_output).item(),
        "norm_mean": np.mean(norms),
        "norm_std": np.std(norms),
        "num_diff_mean": np.mean(num_dif),
        "num_diff_std": np.std(num_dif),
        "errs_mean": np.mean(errs),
        "errs_std": np.std(errs),
        "norm_tstat": norm_tstat,
        "norm_pvalue": norm_pvalue,
    }


def get_dataframe(name: str) -> pd.DataFrame | pd.Series:
    """Get dataframe from pickle file.

    Parameters
    ----------
    name
        The file name for the pickled data.

    Returns
    -------
    data
        A data frame containing the pickled data if it already exists;
        otherwise an empty data frame.
    """
    data_file = f"data/{name}.pkl"

    if pathlib.Path(data_file).is_file():
        data = pd.read_pickle(data_file)
    else:
        data = pd.DataFrame()

    return data


def nn_benchmark(
    params_loop,
    dim_loop,
    kernel_loop_func,
    kernel_name,
    iterations,
    backward: bool = False,
):
    print("----------Benchmarking {}----------".format(kernel_name))
    data = get_dataframe(kernel_name.__name__)
    for model, input, hyper_params, data_params in kernel_loop_func(
        kernel_name, params_loop, dim_loop
    ):
        if data.shape[0] > 0:
            completed = set(
                tuple(row)
                for row in data[
                    [
                        *(
                            list(hyper_params.asdict().keys())
                            + list(data_params.asdict().keys())
                        )
                    ]
                ].to_records(index=False)
            )
        else:
            completed = set()
        if (
            tuple(hyper_params.asdict().values()) + tuple(data_params.asdict().values())
            in completed
        ) and set(err_metrics()).issubset(list(data.columns.values)):
            continue

        base_output, outputs = nn_get_data(
            model, input, iterations=iterations, backward=backward
        )
        error_metrics = all_error_metrics(base_output, outputs)
        latency_metrics = nn_latency(model, input, iterations=iterations)
        new_row = pd.DataFrame(
            [
                hyper_params.asdict()
                | data_params.asdict()
                | error_metrics
                | latency_metrics
                | {"iterations": iterations}
            ]
        )
        data = pd.concat([data, new_row], ignore_index=True)

    # Create path to data file and make `data` directory if needed
    # Then write pickled data to the file
    data_file = f"data/{kernel_name.__name__}.pkl"
    pathlib.Path(data_file).parent.mkdir(parents=True, exist_ok=True)

    data.to_pickle(data_file)


def func_benchmark(params_loop, dim_loop, kernel_loop_func, kernel_name, iterations):
    print("----------Benchmarking {}----------".format(kernel_name))
    data = get_dataframe(kernel_name)
    for model, input, hyper_params, data_params in kernel_loop_func(
        kernel_name, params_loop, dim_loop
    ):
        if data.shape[0] > 0:
            completed = set(
                tuple(row)
                for row in data[
                    [
                        *(
                            list(hyper_params.asdict().keys())
                            + list(data_params.asdict().keys())
                        )
                    ]
                ].to_records(index=False)
            )
        else:
            completed = set()
        if (
            tuple(hyper_params.asdict().values()) + tuple(data_params.asdict().values())
            in completed
        ) and set(err_metrics()).issubset(list(data.columns.values)):
            continue
        base_output, outputs = func_get_data(model, input, iterations=iterations)
        error_metrics = all_error_metrics(base_output, outputs)
        latency_metrics = func_latency(model, input, iterations=iterations)
        new_row = pd.DataFrame(
            [
                hyper_params.asdict()
                | data_params.asdict()
                | error_metrics
                | latency_metrics
            ]
        )
        data = pd.concat([data, new_row], ignore_index=True)

    # Create path to data file and make `data` directory if needed
    # Then write pickled data to the file
    data_file = f"data/{kernel_name}.pkl"
    pathlib.Path(data_file).parent.mkdir(parents=True, exist_ok=True)

    data.to_pickle(data_file)
