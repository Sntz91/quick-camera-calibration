from calculate_statistics import read_data, filter_bad_hulls
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def main():
    # 1. Get Data
    img = 'DJI_0026'
    data_full = get_data(img)
    data_good = get_data(img, quality='good')
    # data_bad = get_data(img, quality='bad')
    # Q1: Is there a difference over ks?
    # a) Box plot
    # create_boxplots('out/boxplots', data_full, data_good)

    # b) P-Value Differences
    # print_p_value_overview(data_full, data_good)

    # c) Histograms
    make_hists(data_full, data_good)

    # Q2: Is there a difference within each k?
    # Already answered?


def get_data(img_name, quality=None):
    raw_data = read_data(img_name)
    raw_data = filter_bad_hulls(raw_data)
    if quality:
        raw_data = raw_data.loc[raw_data['quality'] == quality]
    data_per_experiment = (raw_data.groupby(['comb', 'exp'])
                           .agg(mean_error=('error', 'mean'))
                           .sort_values(by='mean_error').reset_index())
    # TODO GOOD BAD FILTER.

    data_k4 = (data_per_experiment
               .loc[data_per_experiment['comb'] == 4]['mean_error']
               .to_numpy())
    data_k5 = (data_per_experiment
               .loc[data_per_experiment['comb'] == 5]['mean_error']
               .to_numpy())
    data_k6 = (data_per_experiment
               .loc[data_per_experiment['comb'] == 6]['mean_error']
               .to_numpy())
    data_k7 = (data_per_experiment
               .loc[data_per_experiment['comb'] == 7]['mean_error']
               .to_numpy())
    data_k8 = (data_per_experiment
               .loc[data_per_experiment['comb'] == 8]['mean_error']
               .to_numpy())
    data_k9 = (data_per_experiment
               .loc[data_per_experiment['comb'] == 9]['mean_error']
               .to_numpy())
    return data_k4, data_k5, data_k6, data_k7, data_k8, data_k9


def bootstrap(data_A, data_B, N=100000, fn=np.mean, abs=True):
    overall = np.concatenate((data_A, data_B), axis=None)
    if abs:
        obs_diff = np.abs(fn(data_A) - fn(data_B))
    else:
        obs_diff = fn(data_A) - fn(data_B)
    sampled_diffs = []

    for _ in range(N):
        sample = np.random.choice(overall, size=len(overall), replace=True)
        sample_A = sample[:len(data_A)]
        sample_B = sample[len(data_B):]
        if abs:
            sampled_diffs.append(np.abs(fn(sample_A) - fn(sample_B)))
        else:
            sampled_diffs.append(fn(sample_A) - fn(sample_B))

    return sampled_diffs, obs_diff


def create_boxplot(ax, data, outliers=False):
    labels = ['4', '5', '6', '7', '8', '9']
    ax.boxplot(
        data,
        labels=labels,
        showfliers=outliers
    )
    #ax.xlabel('k')
    #ax.ylabel('mean error per experiment')
    return ax


def create_boxplots(filename, data_full, data_good):
    fig, axs = plt.subplots(2, 2)
    axs[0, 0] = create_boxplot(axs[0, 0], data_full, outliers=True)
    axs[1, 0] = create_boxplot(axs[1, 0], data_good, outliers=True)
    axs[0, 1] = create_boxplot(axs[0, 1], data_full, outliers=False)
    axs[1, 1] = create_boxplot(axs[1, 1], data_good, outliers=False)
    axs[0, 0].set_title("Outlier")
    axs[0, 1].set_title("No Outlier")
    axs[0, 0].set(ylabel="Overall")
    axs[1, 0].set(ylabel="Good")
    plt.show()
    fig.savefig(filename, dpi=300)


def print_stat(data_full, data_good, fn=np.var):
    print("Overall")
    for i in range(6):
        print(f"k={4+i} {fn(data_full[i]):.2f}")
    print("Good")
    for i in range(6):
        print(f"k={4+i} {fn(data_good[i]):.2f}")


def print_p_values(data, N=100000, fn=np.mean):
    for i in range(5):
        fake_m_difs, m_dif = bootstrap(data[i], data[i+1], N, fn)
        p_value = np.mean(fake_m_difs >= m_dif)
        print(f"k {i+4}->{i+5}: p_value={p_value*100:.4f} [%]")


def print_p_value_overview(data_full, data_good):
    print("--- P-Values wrt Mean ---")
    print("Overall")
    print_p_values(data_full, fn=np.mean)
    print("Good")
    print_p_values(data_good, fn=np.mean)
    print()
    print("--- P-Values wrt Var ---")
    print("Overall")
    print_p_values(data_full, fn=np.var)
    print("Good")
    print_p_values(data_good, fn=np.var)


def make_hist_individual(filename, data, k=4, fn=np.mean, fn_name='Means'):
    fake_m_difs, m_dif = bootstrap(data[k-4], data[k-3], fn=fn, abs=False)
    left = np.percentile(fake_m_difs, 2.5)
    right = np.percentile(fake_m_difs, 97.5)
    fig, ax = plt.subplots()

    ax.hist(fake_m_difs, color='tomato', edgecolor='black', bins=30, density=True)
    ylim = ax.get_ylim()
    ax.set_title(f"Difference in {fn_name} between {k} and {k+1}")
    y_max = ax.get_ylim()[1]
    ptch = Rectangle((left, 0), right-left, 0.02*y_max, color='greenyellow')
    ax.add_patch(ptch)
    ax.axvline(m_dif, color='black', linestyle='--')
    ax.text(m_dif-0.35*m_dif, 0.8*ylim[1], 'observation', bbox=dict(boxstyle="square", facecolor='white', edgecolor='none', pad=0.1))
    ax.minorticks_on()
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.clf()
    plt.close()


def make_hist(ax, data, k=4, fn=np.mean):
    fake_m_difs, m_dif = bootstrap(data[k-4], data[k-3], fn=fn, abs=False)
    left = np.percentile(fake_m_difs, 2.5)
    right = np.percentile(fake_m_difs, 97.5)

    ax.hist(fake_m_difs, color='tomato', edgecolor='black', bins=30, density=True)
    ylim = ax.get_ylim()
    y_max = ax.get_ylim()[1]
    ptch = Rectangle((left, 0), right-left, 0.02*y_max, color='greenyellow')
    ax.add_patch(ptch)
    ax.axvline(m_dif, color='black', linestyle='--', lw=1)
    # ax.text(m_dif-0.35*m_dif, 0.8*ylim[1], 'observation', bbox=dict(boxstyle="square", facecolor='white', edgecolor='none', pad=0.1))
    ax.minorticks_on()
    #ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    return ax


def make_hists_individually(data_full, data_good):
    make_hist('out/hist_mean_k4_overall', data_full, k=4, fn=np.mean, fn_name='Means')
    make_hist('out/hist_mean_k5_overall', data_full, k=5, fn=np.mean, fn_name='Means')
    make_hist('out/hist_mean_k6_overall', data_full, k=6, fn=np.mean, fn_name='Means')
    make_hist('out/hist_mean_k7_overall', data_full, k=7, fn=np.mean, fn_name='Means')
    make_hist('out/hist_mean_k8_overall', data_full, k=8, fn=np.mean, fn_name='Means')

    make_hist('out/hist_mean_k4_good', data_good, k=4, fn=np.mean, fn_name='Means')
    make_hist('out/hist_mean_k5_good', data_good, k=5, fn=np.mean, fn_name='Means')
    make_hist('out/hist_mean_k6_good', data_good, k=6, fn=np.mean, fn_name='Means')
    make_hist('out/hist_mean_k7_good', data_good, k=7, fn=np.mean, fn_name='Means')
    make_hist('out/hist_mean_k8_good', data_good, k=8, fn=np.mean, fn_name='Means')

    make_hist('out/hist_var_k4_overall', data_full, k=4, fn=np.var, fn_name='Variances')
    make_hist('out/hist_var_k5_overall', data_full, k=5, fn=np.var, fn_name='Variances')
    make_hist('out/hist_var_k6_overall', data_full, k=6, fn=np.var, fn_name='Variances')
    make_hist('out/hist_var_k7_overall', data_full, k=7, fn=np.var, fn_name='Variances')
    make_hist('out/hist_var_k8_overall', data_full, k=8, fn=np.var, fn_name='Variances')

    make_hist('out/hist_var_k4_good', data_good, k=4, fn=np.var, fn_name='Variances')
    make_hist('out/hist_var_k5_good', data_good, k=5, fn=np.var, fn_name='Variances')
    make_hist('out/hist_var_k6_good', data_good, k=6, fn=np.var, fn_name='Variances')
    make_hist('out/hist_var_k7_good', data_good, k=7, fn=np.var, fn_name='Variances')
    make_hist('out/hist_var_k8_good', data_good, k=8, fn=np.var, fn_name='Variances')


def make_hists(data_full, data_good):
    for i in range(5):
        fig, axs = plt.subplots(2, 2)
        fig.suptitle(f'Differences in k={i+4} and k={i+5}')
        axs[0, 0] = make_hist(axs[0, 0], data_full, k=i+4, fn=np.mean)
        axs[1, 0] = make_hist(axs[1, 0], data_good, k=i+4, fn=np.mean)
        axs[0, 1] = make_hist(axs[0, 1], data_full, k=i+4, fn=np.var)
        axs[1, 1] = make_hist(axs[1, 1], data_good, k=i+4, fn=np.var)
        axs[0, 0].set_title("Means")
        axs[0, 1].set_title("Variances")
        axs[0, 0].set(ylabel="Overall")
        axs[1, 0].set(ylabel="Good")
        plt.savefig(f'out/hist_k{i+4}', dpi=300, bbox_inches='tight')
        plt.clf()


if __name__ == '__main__':
    main()
