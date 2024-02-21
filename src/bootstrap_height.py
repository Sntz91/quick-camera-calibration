from calculate_statistics_composition import read_data, filter_bad_hulls
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import stats
import cv2
plt.style.use('dark_background')


def main():
    # 1. Get Data
    #data_full_26 = get_data('DJI_0026')
    #data_full_29 = get_data('DJI_0029')

    #get_stat_html_table(
    #    experiment_dict,
    #    fn=np.mean
    #)
    # TODO the functions only in one file ofc.
    heights = {
        'DJI_0026': 10.3,
        'DJI_0029': 10.3,
        'DJI_0032': 10.3,
        'DJI_0035': 10.4,
        'DJI_0038': 10.3,
        'DJI_0045': 3.2,
        'DJI_0049': 10.3,
        'DJI_0053': 2.8,
        'DJI_0061': 10.1,
        'DJI_0066': 7.9,
        'DJI_0067': 7.9,
        'DJI_0078': 10.3,
    }
    angles = {
        'DJI_0026': -43.0,
        'DJI_0029': -43.0,
        'DJI_0032': -43.0,
        'DJI_0035': -43.0,
        'DJI_0038': -38.1,
        'DJI_0045': -27.8,
        'DJI_0049': -38.5,
        'DJI_0053': -32.1,
        'DJI_0061': -43.5,
        'DJI_0066': -38.7,
        'DJI_0067': -47.3,
        'DJI_0078': -36.8,
    }

    # TODO: Can I get a plot of mean error per Height?
    images = ['DJI_0029', 'DJI_0032', 'DJI_0035', 'DJI_0038', 'DJI_0045', 'DJI_0049', 'DJI_0053', 'DJI_0061', 'DJI_0066', 'DJI_0067', 'DJI_0078']
    raw_data = read_data('DJI_0026')
    for image in images:
        raw_data = pd.concat([raw_data, read_data(image)])
    raw_data = filter_bad_hulls(raw_data)
    data_per_experiment = (raw_data.groupby(['img', 'comb', 'exp'])
                           .agg(mean_error=('error', 'mean'))
                           .sort_values(by='mean_error').reset_index())
    # data_per_image = data_per_experiment.groupby(['img', 'comb'])['mean_error'].mean().reset_index()
    data_per_experiment['heights'] = data_per_experiment.apply(lambda row: heights[row.img], axis=1)
    data_per_experiment['angles'] = data_per_experiment.apply(lambda row: angles[row.img], axis=1)
    print(data_per_experiment)

    for k in range(4, 7):
        data = data_per_experiment.loc[data_per_experiment['comb'] == k]
        x = data['angles'].to_numpy()
        y = data['mean_error'].to_numpy()
        plt.boxplot(
            [[x], [y]]
        )
        # slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
        # line = slope*x+intercept
        # plt.scatter(x, y)
        # plt.plot(x, line)
        plt.show()
        plt.clf()


    # data_good = get_data(img, quality='good')
    # data_bad = get_data(img, quality='bad')
    # Q1: Is there a difference over ks?
    # a) Box plot
    # create_boxplots('out/boxplots', data_full, data_good)

    # b) P-Value Differences
    # print_p_value_overview(data_full, data_good)

    # c) Histograms
    # make_hists(data_full, data_good)

    # Q2: Is there a difference within each k?
    # Already answered?

    # print_stat(data_full, data_good, np.var)


def get_experimental_data(img_name, quality=None):
    raw_data = read_data(img_name)
    raw_data = filter_bad_hulls(raw_data)
    if quality:
        raw_data = raw_data.loc[raw_data['quality'] == quality]
    data_per_experiment = (raw_data.groupby(['comb', 'exp'])
                           .agg(mean_error=('error', 'mean'))
                           .sort_values(by='mean_error').reset_index())
    # print(data_per_experiment)
    data = []
    for k in sorted(data_per_experiment['comb'].unique()):
        data_ = (data_per_experiment
                 .loc[data_per_experiment['comb'] == k]['mean_error']
                 .to_numpy())
        data.append(data_)

    return data

def get_best_k_experimental_data(img_name, k, quality=None):
    raw_data = read_data(img_name)
    raw_data = filter_bad_hulls(raw_data)
    if quality:
        raw_data = raw_data.loc[raw_data['quality'] == quality]
    raw_data = raw_data.loc[raw_data['comb'] == k]
    data_per_experiment = (raw_data.groupby(['comb', 'exp'])
                           .agg(mean_error=('error', 'mean'))
                           .sort_values(by='mean_error').reset_index())
    best = data_per_experiment.loc[data_per_experiment['mean_error'].idxmin()]
    experiment_data = raw_data.loc[raw_data['exp'] == int(best['exp'])]
    return experiment_data['error'].to_numpy()


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


def print_stat(*data_arrs, fn=np.var):
    for data in data_arrs:
        print()
        for i in range(len(data)):
            print(f"k={4+i} {fn(data[i]):.2f}")


def get_stat_html_table(exp_dict, fn=np.mean):
    print('<table class="dataframe">')
    for key, data in exp_dict.items():
        print("<tr>")
        print(f"<th>{key}</th>")
        for i in range(len(data)):
            print("<td>", end="")
            print(f"k={4+i} {fn(data[i]):.2f}", end="")
            print("</td>")
    print("</table>")


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


def make_hist(ax, data, k=4, fn=np.mean):
    fake_m_difs, m_dif = bootstrap(data[k-4], data[k-3], fn=fn, abs=False)
    left = np.percentile(fake_m_difs, 2.5)
    right = np.percentile(fake_m_difs, 97.5)

    ax.hist(fake_m_difs, color='tomato', edgecolor='#303030', bins=30, density=True)
    y_max = ax.get_ylim()[1]
    ptch = Rectangle((left, 0), right-left, 0.03*y_max, color='SlateBlue')
    ax.add_patch(ptch)
    ax.axvline(m_dif, color='white', linestyle='--', lw=1)
    ax.minorticks_on()
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    return ax


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
        plt.savefig(f'out/hist_k{i+4}_dark', dpi=300, bbox_inches='tight')
        plt.clf()

def plot_good_bad_perspectives():
    files = ['DJI_0026', 'DJI_0029', 'DJI_0045', 'DJI_0053']
    imgs = []
    for img_name in files:
        fname = f'/home/ziegleto/ziegleto/data/5Safe/vup/homography_evaluation/data/perspective_views/{img_name}.JPG'
        img = cv2.imread(fname)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)

    fig, axs = plt.subplots(2, 2)
    for i, img in enumerate(imgs):
        ax = axs[int(i > 1), i % 2]
        ax.axis('off')
        ax.imshow(img)
    plt.tight_layout()
    plt.savefig('out/bad_good_perspectives', dpi=300)


if __name__ == '__main__':
    # main()
    k=5
    data_full_26 = get_best_k_experimental_data('DJI_0026', k)
    data_full_29 = get_best_k_experimental_data('DJI_0029', k)
    data_full_45 = get_best_k_experimental_data('DJI_0045', k)
    data_full_53 = get_best_k_experimental_data('DJI_0053', k)
    print(np.mean(data_full_26))
    print(np.mean(data_full_29))
    print(np.mean(data_full_45))
    print(np.mean(data_full_53))

    good_perspective = np.concatenate([data_full_26, data_full_29])
    bad_perspective = np.concatenate([data_full_45, data_full_53])

    # fake_m_difs, m_dif = bootstrap(good_perspective, bad_perspective)
    # p_value = np.mean(fake_m_difs >= m_dif)
    # print(f"p_value={p_value*100:.4f} [%]")

    fig, ax = plt.subplots(1, 1)
    make_hist(ax, [good_perspective, bad_perspective])
    plt.show()
