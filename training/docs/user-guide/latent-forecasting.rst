Latent forecasting
==================

This opt-in training method learns a reconstructable, forecastable latent state,
then permits independent decoder and processor refinement. It uses the normal
Anemoi trainer, task, data indices, preprocessing, distributed strategy and
checkpoint pipeline. It does not replace existing training methods.

Read ``models/docs/usage/latent_forecaster.rst`` for model operations and forcing
conditioning. LUMI-specific paths/resources are documented in
``aila-training/batch_jobs/latent-forecaster/README.md``; they do not belong in
portable core model defaults.

Objectives and gradient flow
----------------------------

With ``z_t = E(x_t)`` and ``SG`` denoting stop-gradient, the stages are:

.. math::

   L_1 &= w_r L_{phys}(D(z_t), x_t)
          + w_z L_{latent}(P(z_t, f_{t+\Delta t}), SG(E(x_{t+\Delta t}))) \\
   L_{2a} &= L_{phys}(D(E_{frozen}(x_t)), x_t) \\
   L_{2b} &= \sum_{k\in S}\bar{w}_k
       L_{latent}(P^{(k)}(E_{frozen}(x_t)), SG(E_{frozen}(x_{t+k\Delta t})))

Stage 1 trains the source encoder, decoder, processor and forcing embeddings.
The target encoder branch runs without gradients and temporarily in evaluation
mode. Stop-gradient does not freeze the source encoder and is not an EMA target
network. Reconstruction protects information retention; latent prediction
supervises forecastability. Neither term alone establishes perfect retention or
long-range stability.

Physical reconstruction uses the configured upstream loss and variable/node
scalers. Diagnostic outputs are supervised against their corresponding targets,
although they are not encoder inputs. Latent error is FP32 squared error averaged
over hidden nodes and channels, with model-group reduction. It does not inherit
physical-variable weights. Positive-lead weights are normalised by their sum.

Stage 2a trains decoder-private parameters only. Stage 2b trains the processor
and its private forcing embeddings, with the complete codec representation
frozen. Both can start independently from the same stage-1 checkpoint. Stage 2b
still executes the frozen encoder for the initial state and supervised target
states; freezing removes encoder backpropagation, not forward cost.

Task and loss configuration
---------------------------

.. code-block:: yaml

   task:
     _target_: anemoi.training.tasks.latent_forecast.LatentForecast
     timestep: 6h
     forecast_steps: 1
     loss_steps: [0, 1]
   training:
     method:
       _target_: anemoi.training.train.methods.latent_forecast.LatentForecastTraining
     latent:
       mode: joint
       reconstruction_weight: 1.0
       latent_weight: 1.0
       lead_weights: [1.0]
       validation_decode_steps: [1]

These are configuration fragments, not standalone training files. The packaged
``latent_forecaster.yaml`` provides the other model/data/trainer settings.

``task.timestep`` is the processor interval. ``data.frequency`` controls dataset
sampling; they need not be equal, but required target timestamps must exist.
``forecast_steps`` is the number of autoregressive transitions. ``loss_steps``
is a sorted, unique list of integer offsets between zero and that horizon.
Zero denotes current-time reconstruction; positive offsets select latent targets.
``lead_weights`` has exactly one nonnegative weight per positive loss step, with
positive sum. The scalar reconstruction/latent weights are applied separately.

All intermediate transitions execute even when their losses are omitted. The
task loads prescribed forcings at every transition time and masks future
prognostic inputs. Only the separate target-encoding branch sees future truth;
it is never fed into the predicted rollout.

For ``mode: decoder``, set ``forecast_steps: 0``, ``loss_steps: [0]``,
``lead_weights: []``, ``validation_decode_steps: []``, reconstruction weight one
and latent weight zero. No processor or forcing embedding execution is needed.
For ``mode: rollout``, set reconstruction weight zero and latent weight one.

``validation_decode_steps`` selects positive leads decoded for physical
validation metrics only. It must fit within the horizon. These physical metrics
do not contribute to the latent training objective. Training-time decoding is
omitted in rollout mode. The current supplied configurations disable plot
callbacks; they do not provide a dedicated latent-task plotting adapter.

Rollout configuration
---------------------

Rollout length and supervised leads are fixed YAML settings for each run; this
method does not implement an automatic curriculum. When changing the horizon,
keep ``forecast_steps``, ``loss_steps``, ``lead_weights``,
``validation_decode_steps`` and model ``decode_steps`` consistent. Prediction
training requires at least one positive supervised lead. Checkpoint continuation
uses the upstream loading mechanisms described below.

Checkpoint transfer and freezing
--------------------------------

Refinement uses the upstream checkpoint pipeline: a source, loader and modifiers.
Cold start loads model weights and resets training progress/optimiser/scheduler.
Warm start resumes training state. These have different purposes.

The stage-2b processor may differ from stage 1. The supplied configuration uses
``ColdStartLoader(strict=False, skip_mismatched=True)``: compatible tensors load,
shape-mismatched MLP tensors and missing additional blocks remain initialised.
This does not widen an MLP while preserving its complete old function.

After loading, ``VerifyLatentRepresentation`` requires exact restoration of
encoder/shared-node/encoder-provider/preprocessing tensors and checks a
``latent_representation_id`` stored in the source checkpoint. The fingerprint
also covers representation-related configuration, ordered variables, graph
coordinates and encoder connectivity. Execution-only encoder settings
(``num_chunks``, ``gradient_checkpointing`` and ``cpu_offload``) are excluded.
It is not an archive manifest or a check
of the source dataset contents. Missing or changed representation identities
fail the refinement load, even with a permissive processor loader.

Use the upstream ``FreezingModifierStage`` with explicit module paths. Required
frozen representation modules are:

.. code-block:: yaml

   submodules_to_freeze:
     - model.model.encoder
     - model.model.encoder_graph_provider
     - model.model.node_attributes
     - model.model.latent_aggregator
     - model.pre_processors
     - model.post_processors

Stage 2a additionally freezes ``model.model.processor``,
``model.model.processor_graph_provider``, ``model.model.static_embeddings`` and
``model.model.temporal_embeddings``. Stage 2b additionally freezes
``model.model.decoder`` and ``model.model.decoder_graph_provider``.
The training method checks these freezing requirements at fit start.

Parallel outputs must eventually be assembled with a matching representation
identity: decoder from 2a and processor/forcing embeddings from 2b. A dedicated
assembly tool is not implemented. Do not resume one branch's optimiser state
into the other branch.

Metrics and scheduler
---------------------

Validation reports ``reconstruction``, ``latent_prediction`` and
``latent_channel_std``, plus configured physical metrics and selected decoded
validation leads prefixed ``forecast_step_<k>_``. Latent spread is a monitoring
statistic, not a variance penalty. Training logs the total weighted loss;
the additional spread reductions run only during validation. Latent MSE alone is insufficient verification;
inspect decoded meteorological skill, small-scale detail and rollout stability.

The LUMI preset uses ReduceLROnPlateau on ``val_multi_dataset_loss``. Its monitored
quantity is the weighted joint objective in stage 1, reconstruction in 2a, and
weighted latent rollout error in 2b. It is not decoded forecast RMSE. Patience
counts consecutive validation checks without a qualifying improvement. No moving
average is applied. The supplied epoch integration, factor 0.8, patience 30,
relative threshold 0.001 and absolute effective ``min_lr`` are YAML settings.
Changing supervised leads changes the monitored objective, which matters for
continuing plateau-scheduler history.

Data semantics and operational adaptation
-----------------------------------------

Decoder-only ``tp``, ``ssrd`` and ``strd`` can retain six-hour accumulation windows
while a processor advances three hours. Successive predictions then have
overlapping windows, not three-hour totals. Do not sum them as disjoint totals
or claim exact shorter accumulations can be recovered from this store.

The current recipes train on ERA5. Operational-analysis adaptation is a later
experiment. Evaluate reconstruction on that distribution before deciding whether
the encoder can remain frozen. If it must change, adapt the representation first
and then refit dependent processors/decoders; regenerate any cached latents.
Freezing the ERA5 encoder throughout operational adaptation is not an agreed
requirement. Physical-space forecast fine-tuning through a decoder and
meteorological variable weights is planned, not implemented as a new stage here.

Implemented boundaries and verification
---------------------------------------

``latent_rollout_loss(initial, targets, conditions, ...)`` accepts already encoded
states and contains no encoder/decoder calls. It leaves an entry point for future
precomputed-latent training. There is no latent-export job, latent archive
backend, cached-data YAML mode, automatic curriculum or global–LAM coordinator.

Focused tests cover model operations/forcing reuse, gradient separation and the
three-stage synthetic CPU train/save/reload pipeline, including a larger and
wider stage-2 processor and exact frozen representation preservation. A separate
two-rank CPU experiment has checked sharded model forward/backward agreement.
A full n320 smoke test on one LUMI node with eight GPUs per model also completed
all three stages, with two optimisation steps per stage, validation and checkpoint
saving. It used the configured codec depths, the 24-block/ratio-6 rollout
processor and no activation checkpointing (job 23149606, 25 October 2026).
This establishes short-run functionality and memory feasibility on that node,
not long-run numerical stability, forecast skill or 128-node throughput.
Representative upstream regression checks passed; this is not exhaustive proof
of compatibility with every external configuration.

Source entry points:

* ``anemoi.models.models.latent_forecaster.AnemoiModelLatentForecaster``
* ``anemoi.training.tasks.latent_forecast.LatentForecast``
* ``anemoi.training.train.methods.latent_forecast.LatentForecastTraining``
* ``anemoi.training.checkpoint.modifiers.latent.VerifyLatentRepresentation``
