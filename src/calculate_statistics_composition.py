import cv2
import re
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
import scienceplots
from stat_data import (
    create_data,
    read_data,
    filter_bad_hulls
)

def overall_results():
    images = ['DJI_0026', 'DJI_0029', 'DJI_0032', 'DJI_0035', 'DJI_0038', 'DJI_0045', 'DJI_0049', 'DJI_0053', 'DJI_0061', 'DJI_0066', 'DJI_0067', 'DJI_0078']
    for img in images:
        df = read_data(img) 
        max_val = df.loc[df['comb'] == max(df['comb'])]
        # print(max_val)
        print(img)
        print('---------')
        print(f'mean error: {np.mean(max_val["error"]):.2f}')
        print(f'var error: {np.var(max_val["error"]):.2f}')
        print(f'sample size: {len(max_val["error"])}')
        # print(max_val)
        print()

def plot_worst_best():
    img = 'DJI_0026'
    df = read_data(img)
    df = filter_bad_hulls(df)

    fig, axs = plt.subplots(5, 8)
    fig.set_size_inches(20, 9.5)
    # Bests
    for k in range(4, 9):
        bests = get_n_best_experiments(df.loc[df['comb']==k], 4)

        ax = axs[k-4, 0]
        best = bests.iloc[0]
        plot_experiment_ax(ax, df, best['comb'], best['exp'])

        ax = axs[k-4, 1]
        best = bests.iloc[1]
        plot_experiment_ax(ax, df, best['comb'], best['exp'])

        ax = axs[k-4, 2]
        best = bests.iloc[2]
        plot_experiment_ax(ax, df, best['comb'], best['exp'])

        ax = axs[k-4, 3]
        best = bests.iloc[3]
        plot_experiment_ax(ax, df, best['comb'], best['exp'])

    plt.tight_layout()

    # Worsts
    for k in range(4, 9):
        worsts = get_n_worst_experiments(df.loc[df['comb']==k], 4)

        ax = axs[k-4, 4]
        best = worsts.iloc[0]
        plot_experiment_ax(ax, df, best['comb'], best['exp'])

        ax = axs[k-4, 5]
        best = worsts.iloc[1]
        plot_experiment_ax(ax, df, best['comb'], best['exp'])

        ax = axs[k-4, 6]
        best = worsts.iloc[2]
        plot_experiment_ax(ax, df, best['comb'], best['exp'])

        ax = axs[k-4, 7]
        best = worsts.iloc[3]
        plot_experiment_ax(ax, df, best['comb'], best['exp'])
    
    plt.tight_layout()
    plt.savefig('out/worst_best.png', dpi=300)
        


def hull_len_3():
    # ERROR 3 PT HULL
    img = 'DJI_0029'
    df = read_data(img)
    df = df.loc[df['comb'] == 4]
    g_df = df.loc[df['quality'] == 'good']
    df_bad_hull = df[df.hull_len <= 3]
    df_good_hull = df[df.hull_len > 3]
    g_df_bad_hull = g_df[g_df.hull_len <= 3]
    g_df_good_hull = g_df[g_df.hull_len > 3]

    bad_hull_exps = df_bad_hull.groupby(['comb', 'exp']) \
               .agg(mean_error=('error', 'mean')) \
               .sort_values(by='mean_error').reset_index()
    good_hull_exps = df_good_hull.groupby(['comb', 'exp']) \
               .agg(mean_error=('error', 'mean')) \
               .sort_values(by='mean_error').reset_index()
    bad_hull_results = bad_hull_exps['mean_error'].to_numpy()
    good_hull_results = good_hull_exps['mean_error'].to_numpy()

    g_bad_hull_exps = g_df_bad_hull.groupby(['comb', 'exp']) \
               .agg(mean_error=('error', 'mean')) \
               .sort_values(by='mean_error').reset_index()
    g_good_hull_exps = g_df_good_hull.groupby(['comb', 'exp']) \
               .agg(mean_error=('error', 'mean')) \
               .sort_values(by='mean_error').reset_index()
    g_bad_hull_results = g_bad_hull_exps['mean_error'].to_numpy()
    g_good_hull_results = g_good_hull_exps['mean_error'].to_numpy()

    fig, axs = plt.subplots(2, 1)
    fig.set_size_inches(8, 8)
    axs[0].boxplot(
        [bad_hull_results, good_hull_results], 
        labels=['$n_{vert}=3$', '$n_{vert}=4$'], 
        showfliers=False,
        vert=False,
        patch_artist=True,
        boxprops=dict(facecolor="lightgray"),
        medianprops=dict(color="black", linewidth=1.5),
        widths=0.3
    )
    axs[1].boxplot(
        [g_bad_hull_results, g_good_hull_results], 
        labels=['$n_{vert}=3$', '$n_{vert}=4$'], 
        showfliers=False,
        vert=False,
        patch_artist=True,
        boxprops=dict(facecolor="lightgray"),
        medianprops=dict(color="black", linewidth=1.5),
        widths=0.3
    )
    # axs[0].set_xlim(0, 25)
    # axs[1].set_xlim(0, 25)
    axs[0].set_title("All Validation Points")
    axs[1].set_title("Only Good Validation Points")
    axs[1].set_xlabel('mean error per experiment [cm]')
    # fig.savefig('out/boxplots_hull.png', dpi=300)
    plt.show()

def main():
    hull_len_3() 
    hull_size_error()
    # best_worst_smallers()

    # print(len(df_))
    # worst = get_n_worst_experiments2(df, 10)
    # print(worst)

    # plot_experiment(df, 5, 4, save=False)
    # results = get_result_table(df)
    # print(results)
IMG = 'DJI_0029'

def best_worst_smallers():
    img = IMG
    df = read_data(img)
    df = filter_bad_hulls(df)
    df = df.loc[df['comb'] == 4]
    exps = df.groupby(['comb', 'exp']) \
           .agg(
               mean_error=('error', 'mean'),
               area=('hull_area', 'max'), # only one 
            ) \
           .sort_values(by='mean_error').reset_index()
    areas = exps['area'].values
    # errors = exps['mean_error'].values
    threshold = max(areas) / 2

    exps['smaller'] = exps['area'] <= threshold
    smallers = exps.loc[exps['smaller'] == True].sort_values(by='mean_error')
    losers = smallers.iloc[-10:]
    winners = smallers.iloc[:10]
    print(losers)

    for index, row in losers.iterrows():
        plot_experiment(df, row['comb'], row['exp'], False)
        plt.cla()


    

    



def hull_size_error():
    # ERROR WRT HULL SIZE
    import matplotlib as mpl
    # overall_results()
    # imgs = ['DJI_0029'] #, 'DJI_0029', 'DJI_0032', 'DJI_0035']#, 'DJI_0038', 'DJI_0045', 'DJI_0049', 'DJI_0053', 'DJI_0061', 'DJI_0066', 'DJI_0067', 'DJI_0078']
    imgs = [IMG]
    colors = ['red', 'green', 'yellow']
    colors = mpl.colormaps['Dark2'].colors
    areas = []
    ov_errors = []
    ov_colors = []
    g_areas = []
    g_ov_errors = []
    for i, img in enumerate(imgs):
        print(f'IMG: {img}')
        # Get Data
        df = read_data(img)
        df = filter_bad_hulls(df)
        df = df.loc[df['comb'] == 4]
        g_df = df.loc[df['quality'] == 'good']

        groups = df.groupby(['comb', 'exp']) \
               .agg(
                   mean_error=('error', 'mean'),
                   area=('hull_area', 'max'), # only one 
                ) \
               .sort_values(by='mean_error').reset_index()
        g_groups = g_df.groupby(['comb', 'exp']) \
               .agg(
                   mean_error=('error', 'mean'),
                   area=('hull_area', 'max'), # only one 
                ) \
               .sort_values(by='mean_error').reset_index()
        area = groups['area'].values
        errors = groups['mean_error'].values
        g_area = g_groups['area'].values
        g_errors = g_groups['mean_error'].values
        # plt.scatter(x=groups['area'], y=groups['mean_error'])
        # plt.show()
        # plt.cla()
        areas += area.tolist()
        ov_errors += errors.tolist()
        g_areas += g_area.tolist()
        g_ov_errors += g_errors.tolist()

        # MIN AREA, MAX AREA?

        for _ in range(len(area)):
            ov_colors.append(colors[i])
    
        # Needs Gaussian, linear
        # corr, _ = st.pearsonr(area, errors)
        # print(f'Pearsons correlation: {corr:.3f}')

        # Nonlinear, nongaussian ok, monotonic
        # corr, _ = st.spearmanr(area, errors)
        # print(f'Spearmans correlation: {corr:.3f}')
        # Get rid of worst 
        # corr, _ = st.spearmanr(area[:-30], errors[:-30])
        # print(f'Spearmans correlation: {corr:.3f}')
    areas = np.array(areas)
    g_areas = np.array(g_areas)
    print(f'min area: {min(areas)}')
    print(f'max area: {max(areas)}')
    print(f'half (max) area: {max(areas)/2}')
    threshold = max(areas) / 2
    g_threshold = max(g_areas) / 2
    nr_overall = len(areas)
    biggers = np.argwhere(areas > threshold)
    smallers = np.argwhere(areas <= threshold)
    g_biggers = np.argwhere(g_areas > g_threshold)
    g_smallers = np.argwhere(g_areas <= g_threshold)
    print(f'nr overall {nr_overall}, nr_bigger {len(biggers)}, nr_smaller {len(smallers)}')
    biggers_error = np.take(ov_errors, biggers, 0).squeeze()
    smallers_error = np.take(ov_errors, smallers, 0).squeeze()
    g_biggers_error = np.take(g_ov_errors, g_biggers, 0).squeeze()
    g_smallers_error = np.take(g_ov_errors, g_smallers, 0).squeeze()
    print(biggers_error)


    fig, axs = plt.subplots(2, 1)
    fig.set_size_inches(8, 8)
    # fig.suptitle('Error w/o bad hulls wrt Area', fontsize=16)
    axs[0].scatter(x=areas, y=ov_errors, c='gray', alpha=0.5)
    # axs[0].set_ylabel('mean error per experiment [cm]')
    # axs[0].set_xlabel('spanned area of hull')
    axs[1].scatter(x=g_areas, y=g_ov_errors, c='gray', alpha=0.5)
    axs[1].set_ylabel('mean error per experiment [cm]')
    axs[1].set_xlabel('spanned area of hull')
    axs[0].set_title("All Validation Points")
    axs[1].set_title("Only Good Validation Points")
    # fig.savefig('out/scatter_area.png', dpi=300)
    plt.show()
    # plt.show()
    plt.cla()

    fig, axs = plt.subplots(2, 1)
    fig.set_size_inches(8, 8)
    # fig.suptitle('Error w/o bad hulls wrt Smaller / Bigger', fontsize=16)
    axs[0].boxplot(
        [biggers_error, smallers_error],
        labels=['large', 'small'],
        showfliers=False,
        vert=False,
        patch_artist=True,
        boxprops=dict(facecolor="lightgray"),
        medianprops=dict(color="black", linewidth=1.5)
    )
    axs[1].boxplot(
        [g_biggers_error, g_smallers_error],
        labels=['large', 'small'],
        showfliers=False,
        vert=False,
        patch_artist=True,
        boxprops=dict(facecolor="lightgray"),
        medianprops=dict(color="black", linewidth=1.5)
    )
    # axs[0].set_xlim(0, 65)
    # axs[1].set_xlim(0, 65)
    axs[1].set_xlabel('mean error per experiment [cm]')
    axs[0].set_title("All Validation Points")
    axs[1].set_title("Only Good Validation Points")
    # fig.savefig('out/boxplots_area.png', dpi=300)
    plt.show()

    # WIE SCHAUTS AUS NACHDEM ICH 3 hull weg hab und nur noch biggers?!
    # TODO
    
    # Woher kommen die outlier, denn die smallers sind ja nicht alle schlecht?
    # 1. Plot some of the bad smallers and the good smallers?

    

        # NOT NECESSARILY BUT WHEN OUTLIER THEN SMALL... 

    
    # df = df.loc[df['quality'] == 'good']
    # fig, ax = plt.subplots()
    # fig.set_size_inches(5, 5)
    # plot_experiment_ax(ax, df, 8, 0)
    # plt.tight_layout()
    # plt.savefig(f'out/{img}_results.png', dpi=300)


    # plot_winners_losers(df, 'out/worst_best')
    # for i in range(100):
    # plot_experiment(df, 10, 0, save=True)

    # save_results(df, img)
    # results = get_result_table(df)
    # print(results)
    # winners = get_n_best_experiments(df, 10)
    # print(winners)
    # df = df.loc[df['comb']==13]
    # losers = get_n_worst_experiments(df, 10)
    # print(losers)

def get_n_worst_experiments2(df, n):
    groups = df.groupby(['comb', 'exp', 'hull_len']) \
               .agg(mean_error=('error', 'mean')) \
               .sort_values(by='mean_error').reset_index()
    return groups.iloc[-n:]

def get_n_worst_experiments(df, n):
    groups = df.groupby(['comb', 'exp']) \
               .agg(mean_error=('error', 'mean')) \
               .sort_values(by='mean_error').reset_index()
    return groups.iloc[-n:]


def get_n_best_experiments(df, n):
    groups = df.groupby(['comb', 'exp']) \
               .agg(mean_error=('error', 'mean')) \
               .sort_values(by='mean_error').reset_index()
    return groups.iloc[:n]


def plot_experiment(df, comb, exp, save=True):
    exp_df = df[(df.comb == comb) & (df.exp == exp)]
    _, ax = plt.subplots()
    ax = plot_experiment_ax(ax, df, comb, exp)
    if save:
        plt.savefig(
            f'out/{exp_df.img.values[0]}_{comb}_{exp}',
            dpi=300,
            bbox_inches='tight'
        )
    else:
        plt.show()
    plt.clf()


def plot_experiment_ax(ax, df, comb, exp):
    exp_df = df[(df.comb == comb) & (df.exp == exp)]
    mean_error = exp_df.error.mean()
    fname = f'/home/ziegleto/ziegleto/data/5Safe/vup/homography_evaluation/data/perspective_views/{exp_df.img.values[0]}.JPG'
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Plotting
    ax.imshow(img)

    # Plot Reference Points
    ref_pts = exp_df.ref_pts.values[0]
    ax.scatter(ref_pts[:, 0], ref_pts[:, 1], color='white', alpha=0.4, s=100)

    # Plot Hull
    hull = exp_df.hull.values[0]
    plot_hull(ax, hull)

    # Plot Validation Points
    plot_validation_points(ax, exp_df)

    t = ax.text(220, 100, f'error: {mean_error:.2f}', color='white')
    t.set_bbox(dict(facecolor='black', alpha=0.5, edgecolor='black'))
    ax.axis('off')
    return ax


def plot_validation_points(ax, exp_df):
    for _, row in exp_df.iterrows():
        color = 'green' if row.quality == 'good' else 'red'
        ax.scatter(row.val_pt_x, row.val_pt_y, c=color, s=80, alpha=0.6)
        t = ax.text(row.val_pt_x, row.val_pt_y+50, f'{row.error:.2f}', color='black')
        # t.set_bbox(dict(facecolor='black', alpha=0.5, edgecolor='black'))


def plot_hull(ax, hull):
    _hull = np.concatenate((hull, [hull[0, :]]))
    ax.plot(_hull[:, 0], _hull[:, 1], color='white', lw=2.0, alpha=0.6)


def get_result_table(df):
    df_ = df.copy()
    df_['quality'] = 'overall'

    # Get Results per experiment
    groups1 = df_.groupby(['quality', 'comb', 'exp'])['error'].agg([
        'mean',
        'min',
        'max',
        ("quantile 0.25", lambda x: x.quantile(0.25)),
        ("quantile 0.50", lambda x: x.quantile(0.5)),
        ("quantile 0.75", lambda x: x.quantile(0.75)),
        'var',
        ("nr val pts", "count")
    ])
    groups2 = df.groupby(['quality', 'comb', 'exp'])['error'].agg([
        'mean',
        'min',
        'max',
        ("quantile 0.25", lambda x: x.quantile(0.25)),
        ("quantile 0.50", lambda x: x.quantile(0.5)),
        ("quantile 0.75", lambda x: x.quantile(0.75)),
        'var',
        ("nr val pts", "count")
    ])
    groups = pd.concat([groups1, groups2], axis=0).reset_index()
    # print(groups.groupby(['quality', 'comb'])['count'].max())
    groups = groups.groupby(['quality', 'comb'])['mean'].agg([
        'mean',
        'min',
        'max',
        ("quantile 0.25", lambda x: x.quantile(0.25)),
        ("quantile 0.50", lambda x: x.quantile(0.5)),
        ("quantile 0.75", lambda x: x.quantile(0.75)),
        'var',
        ('nr of experiments', 'count')
    ])
    groups = groups.reset_index()
    return groups


def get_result_after_quality(df):
    groups = df.groupby(['comb', 'quality']).agg(mean_error=('error', 'mean'))
    print(groups)


def save_html(df, name, suffix):
    html_string = df.to_html(index=False)
    html = re.sub(r'<tr.*>', '<tr>', df.to_html().replace('border="1" ', ''))
    with open(f'out/{name}_{suffix}_table.html', 'w') as out:
        out.write(html_string)


def save_results(df, name):
    results1 = get_result_table(df)
    results1.to_csv(f"out/{name}_table_unfiltered.csv")
    save_html(results1, name, 'unfiltered')

    df = filter_bad_hulls(df)
    results2 = get_result_table(df)
    results2.to_csv(f"out/{name}_table_filtered.csv")
    save_html(results2, name, 'filtered')


def plot_winners_losers(df, filename):
    fig, axs = plt.subplots(6, 8)
    fig.set_size_inches(25, 15)
    for k in range(4, 10):
        losers = get_n_worst_experiments(df.loc[df['comb']==k], 4)
        winners = get_n_best_experiments(df.loc[df['comb']==k], 4)

        axs[k-4, 0].axis('on')
        axs[k-4, 0].set_ylabel(f'k={k}')
        axs[k-4, 0].axes.xaxis.set_ticklabels([])
        axs[k-4, 0].axes.yaxis.set_ticklabels([])

        i = 0
        for index, row in winners.iterrows():
            axs[k-4, i].set_title(f'best - {i+1}')
            plot_experiment_ax(axs[k-4, i], df, row['comb'], row['exp'])
            i += 1
        if len(winners) == 1:  # No worst / best
            break
        for index, row in losers.iterrows():
            axs[k-4, i].set_title(f'worst - {i-3}')
            plot_experiment_ax(axs[k-4, i], df, row['comb'], row['exp'])
            i += 1
    plt.tight_layout()
    plt.show()
    # plt.savefig(filename, dpi=300)


if __name__ == '__main__':
    plt.style.use(['science', 'grid'])
    pd.set_option("display.precision", 2)
    pd.set_option("display.max_rows", None)
    main()
