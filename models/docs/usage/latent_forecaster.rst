.. _latent-forecaster-model:

Latent forecaster
=================

``AnemoiModelLatentForecaster`` separates snapshot encoding, latent advancement
and physical decoding. It is an opt-in model; existing encoder–processor–decoder
models retain their own behaviour. One instance currently represents one domain.

Architecture
------------

Let ``x_t`` denote the prognostic fields and ``f_t`` prescribed forcings.
The model computes:

.. math::

   z_t &= N(E(x_t, f_t)) \\
   c_k &= [M_s(W f_s),\ M_\tau(W f_{t+k\Delta t})] \\
   z_{k+1} &= N(z_k + P(z_k, c_{k+1})) \\
   \widehat{x}_k &= D(z_k, f_{t+k\Delta t})

``E`` and ``D`` use upstream graph-transformer mappers. ``P`` uses the upstream
``GraphTransformerProcessor`` and its hidden graph. Attention follows graph
edges; it is not unrestricted attention over every hidden node. The residual
update is selected by ``model.latent_skip``. With it disabled, ``N(P(...))``
replaces the residual expression.

``N`` is per-node channel LayerNorm without learned affine parameters when
``model.latent.normalize_state`` is true. Otherwise it is identity. This controls
latent scale; it does not establish injectivity or prevent every collapsed
representation. Reconstruction and latent-spread diagnostics remain necessary.

Encoder inputs include the configured prognostic and forcing fields. Diagnostic
variables are output-only. Decoder node features may include forcings,
coordinates and learned node attributes, but cannot include ``prognostics`` or
``encoded_data`` shortcuts. The decoder must recover atmospheric information
from the latent state. Outputs include prognostic fields and diagnostics.
Decoder target features use the standard feature components. Forcing features
contain one valid-time snapshot, independently of the rollout length.

Forcing conditioning
--------------------

``W`` is a fixed, inverse-distance, row-normalised mapping along data-to-hidden
edges, applied with upstream ``SparseProjector``. Hidden-node sine/cosine
coordinate features are evaluated directly at hidden coordinates. Other fields,
including calendar and solar forcings, retain the dataset definitions and are
mapped to the hidden grid. No new analytic solar implementation is introduced.

``M_s`` and ``M_tau`` are separately configured upstream pointwise MLPs.
Their concatenated output conditions processor ``ConditionalLayerNorm`` layers.
The processor's ``condition_shape`` must equal the sum of embedding output
widths. Static and temporal variable lists must be disjoint subsets of the
configured input forcings.

Static conditioning is computed once per forward/training rollout and reused.
It is recomputed on subsequent calls so trainable embeddings do not leave stale
cached values. Temporal conditioning is computed at each target valid time.
There are no additional learned spatial forcing encoders.

Independent operations
----------------------

The following operations are exposed on ``interface.model``:

* ``encode_snapshot(x, time_index, ...)`` returns the normalised latent tensor
  and hidden-node shard sizes
* ``static_condition(x, ...)`` and ``condition(x, time_index, static=..., ...)``
  produce local hidden-node conditions
* ``advance(current, condition, ..., step=None, message=None)`` runs one
  processor transition, without encoding or decoding
* ``decode_snapshot(latent, x, time_index, ...)`` decodes a selected state using
  prescribed features at its valid time

Physical inputs use the existing Anemoi five-dimensional layout
``[batch, time, ensemble, data_nodes, input_channels]``. Latents and conditions
use ``[local_hidden_nodes, channels]`` in the current batch-one, ensemble-one
model-parallel configuration. Pass the shard sizes returned by the encoder to
advancement/decoding; do not replace them with the local tensor length.

``step`` checks the trained interval; it does not select a different numerical
step size. ``message`` must match the receiving latent tensor's shape. If
present, it is added to the processor input; the residual base remains the
current state. An absent message and an all-zero message are equivalent.
This is an extension point, not an implemented global–regional exchange model.

The regular ``forward``/``predict_step`` path retains Anemoi physical-output
conventions. It encodes the initial state once, advances through supplied forcing
times, and decodes ``model.latent.decode_steps`` in temporal order. Forcings must
cover those times. No decode/re-encode occurs between processor steps.

Configuration
-------------

The model target is:

.. code-block:: yaml

   model:
     model:
       _target_: anemoi.models.models.latent_forecaster.AnemoiModelLatentForecaster
     latent_skip: true
     latent:
       timestep: ${task.timestep}
       decode_steps: ${task.loss_steps}
       normalize_state: true

A complete model configuration also supplies the standard encoder, decoder,
processor and graph-provider settings, forcing variable lists, embedding MLPs,
and sparse projector. See the packaged ``latent_forecaster.yaml`` training
example and the training guide. Embedding width, processor latent width, mapper
depth, attention heads and MLP ratio are independent YAML choices subject to the
underlying layers' shape constraints.

Stacked encoders and decoders
-----------------------------

Deeper encoders and decoders are opt-in mapper subclasses. Select them through
the standard YAML ``_target_`` mechanism:

.. code-block:: yaml

   encoders:
     '0':
       mapper:
         _target_: anemoi.models.layers.mapper_stack.StackedGraphTransformerForwardMapper
         num_layers: 5
   decoders:
     '0':
       mapper:
         _target_: anemoi.models.layers.mapper_stack.StackedGraphTransformerBackwardMapper
         num_layers: 3

``num_layers`` counts complete, independently parameterised mapper blocks,
following upstream processor terminology. Standard mapper targets
and schemas remain single-block and do not acquire a depth option.

The stacked subclasses inherit upstream preprocessing, forward execution,
sharding, spatial chunking and gradient checkpointing. The base mapper's
``_build_mapper_block`` construction hook returns the original single block;
the subclasses return a ``GraphTransformerMapperStack``. Both paths use the
existing ``self.proc(...)`` calls. CPU offload for the opt-in classes wraps the
stack component.

Each block reads the original embedded source features and the preceding
block's destination features, with the same graph, sharding and conditioning
arguments. This refines the destination representation without introducing
source-to-source graph edges. Parameter names are ``proc.blocks.<index>.*``.
Old experimental multi-block checkpoints are not migrated; existing upstream
single-block configurations and checkpoint paths remain unchanged.

Boundaries of this implementation
---------------------------------

Multi-domain composition, multirate coordination, learned exchange modules,
latent archive generation/loading, parallel-stage checkpoint assembly tooling
and operational inference-runner integration are not delivered by this model.
Independent advancement and representation checking leave these extensions
possible. A global 3-hour/LAM 1-hour system will need a coordinator with explicit
exchange times and a policy for information between global updates; linear
interpolation of learned latents is not assumed valid.

Old experimental predictive-autoencoder checkpoints are not promised compatible
with this new class. Existing upstream model classes remain separate.
