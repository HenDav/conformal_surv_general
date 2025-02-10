import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# A function to plot the average coverage and average LPB of each method for a given target alpha and a given setting. Create two figs, one for coverage and one for LPB. 
# Use boxplots to show the average coverage and LPB for each method. 
def plot_coverage_lpb(df, target_alpha, setting):
    fig, ax = plt.subplots(1, 2, figsize=(21, 6))
    df_plot = df.loc[(slice(None), target_alpha, setting), :].copy()
    df_plot = df_plot.reset_index()
    # Make the dpi 100
    ax.figure.set_dpi(300)
    sns.boxplot(x='Method', y='Coverage', data=df_plot, ax=ax[0])
    sns.boxplot(x='Method', y='LPB', data=df_plot, ax=ax[1])
    ax[0].set_title(f'Average Coverage for Target Alpha {target_alpha} and Setting {setting}')
    ax[1].set_title(f'Average LPB for Target Alpha {target_alpha} and Setting {setting}')
    plt.show()

def plot_lpb(df, target_alpha, setting):
    fig, ax = plt.subplots(1, 1, figsize=(7, 3))
    df_plot = df.loc[(slice(None), target_alpha, setting), :].copy()
    df_plot = df_plot.reset_index()
    # Print non-numeric values
    sns.boxplot(x='Method', y='LPB', data=df_plot, ax=ax, palette='tab10')
    ax.set_ylabel('LPB (months)')
    # y-axis grid
    ax.yaxis.grid(True)
    # set dpi to 300
    ax.figure.set_dpi(300)
    plt.show()

def plot_lpbs(df, target_alpha):
    # Plot all the methods' LPBs for a given target alpha and each setting with sns
    fig, ax = plt.subplots(1, 6, figsize=(7, 3))
    for i, setting in enumerate(df.index.get_level_values('Setting').unique()):
        df_plot = df.loc[(slice(None), target_alpha, setting), :].copy()
        df_plot = df_plot.reset_index()
        sns.boxplot(x='Method', y='LPB', data=df_plot, ax=ax[i], palette='tab10')
        ax[i].set_title(f'{setting}')
        ax[i].set_ylabel('LPB (months)')
        ax[i].yaxis.grid(True)
        # Remove x-axis labels and ticks
        ax[i].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax[i].set_xlabel('')
        # Remove ylabel from all but the first plot
        if i != 0:
            ax[i].set_ylabel('')
        # set dpi to 300
        ax[i].figure.set_dpi(300)
    # Create a legend
    handles = [mpatches.Patch(color=sns.color_palette('tab10')[i], label=method) for i, method in enumerate(df_plot['Method'].unique())]
    fig.legend(handles=handles, loc='center right', ncol=1, bbox_to_anchor=(1.2, 0.5))
    plt.tight_layout()
    plt.show()

def plot_coverage_and_lpb_comperison(df, target_alpha, fontsize=30):
    # Create subplots with 3 rows and 6 columns
    # Adjust the height ratios to give less vertical space to the last row
    fig, axes = plt.subplots(2, 6, figsize=(28, 12))

    df_plot = df.loc[(slice(None), target_alpha), :].copy().reset_index()

    methods = df_plot['Method'].unique()
    palette = dict(zip(methods, sns.color_palette(palette='tab10', n_colors=len(methods))))
    settings = sorted(df_plot['Setting'].unique())
    
    # Compute median LPB of the Naive method for normalization
    naive_median_lpb = df_plot[df_plot['Method'] == 'Naive'].groupby('Setting')['LPB'].median()
    df_plot['Relative_LPB'] = df_plot['LPB'] / df_plot['Setting'].map(naive_median_lpb)

    for i, setting in enumerate(settings):
        data = df_plot[df_plot['Setting'] == setting].copy()

        # First row: Coverage
        ax = axes[0, i]
        sns.boxplot(x='Method', y='Coverage', data=data, ax=ax, palette=palette)
        ax.set_title(f'Setting {setting}', fontsize=fontsize)
        ax.set_ylim(0.7, 1.0)
        ax.axhline(0.9, color='red', linestyle='--')  # Dotted red line
        ax.set_xlabel('')  # Remove x-label

        # Remove x-axis ticks and labels
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        if i == 0:
            ax.set_ylabel('Coverage', fontsize=fontsize) # , rotation=0, labelpad=20
            # Align y-axis label to the left
            ax.yaxis.set_label_coords(-0.325, 0.5)
            # ax.yaxis.label.set_horizontalalignment('right')
        else:
            # Remove y-axis labels and ticks
            ax.set_ylabel('')
            ax.tick_params(axis='y', which='both', left=False, labelleft=False)

        # Det y-tick labels to size fontsize / 1.2
        ax.tick_params(axis='y', labelsize=fontsize / 1.2)

        # Add y grid
        ax.yaxis.grid(True)

        # Second row: Relative LPB
        ax = axes[1, i]

        sns.boxplot(x='Method', y='Relative_LPB', data=data, ax=ax, palette=palette)

        # Set y-axis limits appropriately
        y_min = df_plot['Relative_LPB'].min()
        y_max = df_plot['Relative_LPB'].max()
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)

        ax.set_xlabel('')  # Remove x-label

        # Remove x-axis ticks and labels
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        if i == 0:
            ax.set_ylabel('Relative LPB', fontsize=fontsize,) #  rotation=0, labelpad=20
            # Align y-axis label to the left
            ax.yaxis.set_label_coords(-0.3, 0.5)
            # ax.yaxis.label.set_horizontalalignment('right')
        else:
            # Remove y-axis labels and ticks
            ax.set_ylabel('')
            ax.tick_params(axis='y', which='both', left=False, labelleft=False)

        ax.tick_params(axis='y', labelsize=fontsize / 1.2)

        # Add y grid
        ax.yaxis.grid(True)
        
    # Adjust layout with increased left margin
    plt.tight_layout(rect=[0.15, 0, 1, 0.95], h_pad=0.5)  # Adjust left margin to 0.15

    # Create custom legend with increased font size
    handles = [mpatches.Patch(color=palette[method], label=method) for method in methods]
    legend = fig.legend(handles=handles, loc='center right', ncol=1,  fontsize=fontsize-5, bbox_to_anchor=(1, 0.5))
    plt.show()

def plot_c_ind_ibs_and_dcal_comperison(df_metrics, fontsize=30):
    # Create subplots with 3 rows and 6 columns
    # Adjust the height ratios to give less vertical space to the last row

    df_plot = df_metrics.copy().reset_index()
    settings = sorted(df_plot['Setting'].unique())
    fig, axes = plt.subplots(3, len(settings), figsize=(28, 18))

    methods = df_plot['Method'].unique()
    palette = dict(zip(methods, sns.color_palette(palette='tab10', n_colors=len(methods))))
    
    for i, setting in enumerate(settings):
        data = df_plot[df_plot['Setting'] == setting].copy()

        # First row: C-Index
        ax = axes[0, i]
        sns.boxplot(x='Method', y='C-Index', data=data, ax=ax, palette=palette)
        ax.set_title(f'Setting {setting}' if setting in range(1,7) else f'{setting}', fontsize=fontsize)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel('')  # Remove x-label

        # Remove x-axis ticks and labels
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        if i == 0:
            ax.set_ylabel('C-Index', fontsize=fontsize) # , rotation=0, labelpad=20
            # Align y-axis label to the left
            # ax.yaxis.set_label_coords(-0.26, 0.5)
            # ax.yaxis.label.set_horizontalalignment('right')
        else:
            # Remove y-axis labels and ticks
            ax.set_ylabel('')
            ax.tick_params(axis='y', which='both', left=False, labelleft=False)

        # Det y-tick labels to size fontsize / 1.2
        ax.tick_params(axis='y', labelsize=fontsize / 1.2)

        # Add y grid
        ax.yaxis.grid(True)

        # Second row: IBS
        ax = axes[1, i]

        sns.boxplot(x='Method', y='IBS', data=data, ax=ax, palette=palette)

        # Set y-axis limits appropriately
        y_min = df_plot['IBS'].min()
        y_max = df_plot['IBS'].max()
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.05 * y_range, y_max + 0 * y_range)

        ax.set_xlabel('')  # Remove x-label

        # Remove x-axis ticks and labels
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        if i == 0:
            ax.set_ylabel('IBS', fontsize=fontsize,) #  rotation=0, labelpad=20
            # Align y-axis label to the left
            ax.yaxis.set_label_coords(-0.3, 0.5)
            # ax.yaxis.label.set_horizontalalignment('right')
        else:
            # Remove y-axis labels and ticks
            ax.set_ylabel('')
            ax.tick_params(axis='y', which='both', left=False, labelleft=False)

        ax.tick_params(axis='y', labelsize=fontsize / 1.2)

        # Add y grid
        ax.yaxis.grid(True)

        # Third row: D-Cal
        ax = axes[2, i]

        sns.boxplot(x='Method', y='D-Cal', data=data, ax=ax, palette=palette)

        # Set y-axis limits appropriately
        y_min = df_plot['D-Cal'].min()
        y_max = df_plot['D-Cal'].max()
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)

        ax.set_xlabel('')  # Remove x-label

        # Remove x-axis ticks and labels
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        if i == 0:
            ax.set_ylabel('D-Cal', fontsize=fontsize,) #  rotation=0, labelpad=20
            # Align y-axis label to the left
            ax.yaxis.set_label_coords(-0.3, 0.5)
            # ax.yaxis.label.set_horizontalalignment('right')
        else:
            # Remove y-axis labels and ticks
            ax.set_ylabel('')
            ax.tick_params(axis='y', which='both', left=False, labelleft=False)

        ax.tick_params(axis='y', labelsize=fontsize / 1.2)

        # Add y grid
        ax.yaxis.grid(True)

    # Adjust layout with increased left margin
    plt.tight_layout(rect=[0.15, 0, 1, 0.95], h_pad=0.5)  # Adjust left margin to 0.15

    # Create custom legend with increased font size
    handles = [mpatches.Patch(color=palette[method], label=method) for method in methods]
    legend = fig.legend(handles=handles, loc='center right', ncol=1,  fontsize=fontsize, bbox_to_anchor=(1.17, 0.5))

    plt.show()

def plot_lpbs_ablation(df, target_alpha, fontsize=20):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.patches as mpatches

    # Plot all the methods' LPBs for a given target alpha and each setting with sns
    fig, ax = plt.subplots(2, 4, figsize=(14, 5))  # Adjusted figsize for better visualization
    # ax = ax.ravel()
    for i, dq in enumerate(df.index.get_level_values('Depth Quantiles').unique()):
        for j, dw in enumerate(df.index.get_level_values('Depth Weights').unique()):
            # Filter the DataFrame based on the index levels
            df_plot = df.loc[
                (slice(None), target_alpha, 3, slice(None), dq, dw), :
            ].copy()
            df_plot = df_plot.reset_index()
            palette = dict(zip(df_plot['Method'].unique(), sns.color_palette(palette='tab10', n_colors=len(df_plot['Method'].unique()))))
            sns.boxplot(
                x='Method', y='LPB', data=df_plot, ax=ax[1, i+j*2], palette='tab10'
            )
            sns.boxplot(
                x='Method', y='Coverage', data=df_plot, ax=ax[0, i+j*2], palette='tab10'
            )
            ax[0, i+j*2].set_title(f'{dq} and {dw}', fontsize=fontsize, pad=20)
            ax[1, i+j*2].yaxis.grid(True)
            ax[0, i+j*2].yaxis.grid(True)
            # Remove x-axis labels and ticks
            ax[1, i+j*2].tick_params(
                axis='x', which='both', bottom=False, top=False, labelbottom=False
            )
            ax[0, i+j*2].tick_params(
                axis='x', which='both', bottom=False, top=False, labelbottom=False
            )
            ax[1, i+j*2].tick_params(axis='y', labelsize=fontsize)
            ax[0, i+j*2].tick_params(axis='y', labelsize=fontsize)
            ax[1, i+j*2].set_xlabel('')
            ax[0, i+j*2].set_xlabel('')
            # Remove ylabel from all but the first plot
            if i+j*2 != 0:
                ax[0, i+j*2].set_ylabel('')
                ax[1, i+j*2].set_ylabel('')
            else:
                ax[0, i+j*2].set_ylabel('Coverage', fontsize=fontsize)
                ax[1, i+j*2].set_ylabel('LPB', fontsize=fontsize, labelpad=15)
            ax[1, i+j*2].figure.set_dpi(300)
            ax[0, i+j*2].figure.set_dpi(300)
            # Add a horizontal line at the target alpha value
            ax[0, i+j*2].axhline(y=1-target_alpha, color='r', linestyle='--')
            handles = [mpatches.Patch(color=palette[method], label=method) for method in df_plot['Method'].unique()]

    # Create a legend
    fig.legend(
        handles=handles, loc='center right', ncol=1, bbox_to_anchor=(1.18, 0.5), fontsize=fontsize
    )
    plt.tight_layout()
    plt.show()

# Plot three horizontal linepolots one above the other, each as a function of end of trial time. The first is censorship rate, the second is LPB, and the third is coverage.
def plot_censorship_rate_lpb_coverage(df, target_alpha, fontsize=12):
    # Plot all the methods' LPBs for a given target alpha and each setting with sns
    fig, ax = plt.subplots(3, 1, figsize=(10, 5))  # Adjusted figsize for better visualization
    # ax = ax.ravel()
    # Filter the DataFrame based on the index levels
    df_plot = df.loc[
        (slice(None), target_alpha, 3, slice(None)), :
    ].copy()
    df_plot = df_plot.reset_index()
    palette = dict(zip(df_plot['Method'].unique(), sns.color_palette(palette='tab10', n_colors=len(df_plot['Method'].unique()))))
    sns.lineplot(
        x='End of trial time', y='Censorship Rate', data=df_plot[df_plot['Method']=='Naive'], ax=ax[0], palette='tab10', ci=None
    )
    sns.lineplot(
        x='End of trial time', y='LPB', data=df_plot, ax=ax[2], palette='tab10', hue='Method'
    )
    sns.lineplot(
        x='End of trial time', y='Coverage', data=df_plot, ax=ax[1], palette='tab10', hue='Method'
    )
    ax[2].yaxis.grid(True)
    ax[1].yaxis.grid(True)
    ax[0].yaxis.grid(True)
    # Remove x-axis labels and ticks
    ax[1].tick_params(
        axis='x', which='both', bottom=False, top=False, labelbottom=False
    )
    ax[0].tick_params(
        axis='x', which='both', bottom=False, top=False, labelbottom=False
    )
    ax[1].set_xlabel('')
    ax[0].set_xlabel('')
    ax[1].figure.set_dpi(300)
    ax[0].figure.set_dpi(300)
    # Add a horizontal line at the target alpha value
    ax[1].axhline(y=1-target_alpha, color='r', linestyle='--')

    # Set the fontsize of the y-axis labels
    ax[0].set_ylabel('Censorship\nRate', fontsize=fontsize)
    for i in range(3):
        ax[i].tick_params(axis='y', labelsize=fontsize)
        ax[i].set_ylabel(ax[i].get_ylabel(), fontsize=fontsize)
        # Set the x limits to be the lowest and highest end of trial times
        ax[i].set_xlim(df_plot['End of trial time'].min(), df_plot['End of trial time'].max())
        
    
    # Move the ax[2] y-axis label to the left
    ax[2].yaxis.set_label_coords(-0.095, 0.5)

    # Move the ax[1] y-axis label to the left
    ax[1].yaxis.set_label_coords(-0.095, 0.5)

    # Set the x-axis label and ticks for the last plot to the right font size
    ax[2].set_xlabel('End of trial time', fontsize=fontsize)
    ax[2].tick_params(axis='x', labelsize=fontsize)
    handles = [mpatches.Patch(color=palette[method], label=method) for method in df_plot['Method'].unique()]

    # Create a legend
    fig.legend(
        handles=handles, loc='center right', ncol=1, bbox_to_anchor=(1.2, 0.5), fontsize=fontsize
    )
    # Remove the legend from the second and third plots
    ax[1].get_legend().remove()
    ax[2].get_legend().remove()
    plt.tight_layout()
    plt.show()

def plot_lpb_coverage_n_samples(df, target_alpha, fontsize=12):

    # Plot all the methods' LPBs for a given target alpha and each setting with sns
    fig, ax = plt.subplots(2, 1, figsize=(7, 5))  # Adjusted figsize for better visualization
    # ax = ax.ravel()
    # Filter the DataFrame based on the index levels
    df_plot = df.loc[
        (slice(None), target_alpha, 3, slice(None)), :
    ].copy()
    df_plot = df_plot.reset_index()
    palette = dict(zip(df_plot['Method'].unique(), sns.color_palette(palette='tab10', n_colors=len(df_plot['Method'].unique()))))
    sns.lineplot(
        x='Num Samples', y='LPB', data=df_plot, ax=ax[1], palette='tab10', hue='Method'
    )
    sns.lineplot(
        x='Num Samples', y='Coverage', data=df_plot, ax=ax[0], palette='tab10', hue='Method'
    )
    ax[1].yaxis.grid(True)
    ax[0].yaxis.grid(True)
    # Remove x-axis labels and ticks
    ax[0].tick_params(
        axis='x', which='both', bottom=False, top=False, labelbottom=False
    )
    ax[0].set_xlabel('')
    ax[1].figure.set_dpi(300)
    ax[0].figure.set_dpi(300)
    # Add a horizontal line at the target alpha value
    ax[0].axhline(y=1-target_alpha, color='r', linestyle='--')
    handles = [mpatches.Patch(color=palette[method], label=method) for method in df_plot['Method'].unique()]

    # Create a legend
    fig.legend(
        handles=handles, loc='center right', ncol=1, bbox_to_anchor=(1.2, 0.5), fontsize=fontsize
    )

    # Set the fontsize of the y-axis labels
    ax[0].set_ylabel('Coverage', fontsize=fontsize)
    ax[1].set_ylabel('LPB', fontsize=fontsize, labelpad=5)
    for i in range(2):
        ax[i].tick_params(axis='y', labelsize=fontsize)
        ax[i].set_ylabel(ax[i].get_ylabel(), fontsize=fontsize)
        # Set the x limits to be the lowest and highest end of trial times
        ax[i].set_xlim(df_plot['Num Samples'].min(), df_plot['Num Samples'].max())
        ax[i].tick_params(axis='x', labelsize=fontsize)
    ax[1].set_xlabel('Number of Samples', fontsize=fontsize)

    # Remove the legend from the plots
    ax[0].get_legend().remove()
    ax[1].get_legend().remove()


    plt.tight_layout()
    plt.show()

