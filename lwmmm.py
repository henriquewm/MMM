import jax.numpy as jnp
import numpyro
from lightweight_mmm import lightweight_mmm
from lightweight_mmm import optimize_media
from lightweight_mmm import plot
from lightweight_mmm import preprocessing
from lightweight_mmm import utils
# Let's assume we have the following datasets with the following shapes (we use
#the `simulate_dummy_data` function in utils for this example):
media_data, extra_features, target, costs = utils.simulate_dummy_data(
    data_size=160,
    n_media_channels=3,
    n_extra_features=2,
    geos=5) # Or geos=1 for national model

# Fit model.
mmm = lightweight_mmm.LightweightMMM()
mmm.fit(media=media_data,
        extra_features=extra_features,
        media_prior=costs,
        target=target,
        number_warmup=1000,
        number_samples=1000,
        number_chains=2)

mmm.print_summary()

# %%
plot.plot_model_fit(media_mix_model=mmm)
# %%
plot.plot_response_curves(media_mix_model=mmm)
# %%
media_effect_hat, roi_hat = mmm.get_posterior_metrics()
# %%
roi_hat
# %%
plot.plot_bars_media_metrics(metric=media_effect_hat)
# %%
plot.plot_bars_media_metrics(metric=roi_hat)
# %%
