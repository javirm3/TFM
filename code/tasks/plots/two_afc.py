"""2AFC task-owned plots.

The Alexis / 2AFC plotting API remains implemented in the existing binary-task
plot module, but task code now imports it through ``tasks.plots.two_afc`` so
call sites consistently ask the task for its plots.
"""

from tasks.plots.two_afc_impl import (
    prepare_predictions_df,
    plot_categorical_performance_all,
    plot_categorical_performance_by_state,
    plot_emission_weights,
    plot_model_comparison,
    plot_model_comparison_diffs,
    plot_occupancy,
    plot_occupancy_boxplot,
    plot_posterior_probs,
    plot_session_deepdive,
    plot_session_trajectories,
    plot_state_accuracy,
    plot_state_occupancy,
    plot_trans_mat,
    plot_trans_mat_boxplots,
    plot_weights,
    plot_weights_boxplot,
    plot_weights_per_contrast,
    remap_states,
)
