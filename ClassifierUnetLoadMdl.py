import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
# Requires TensorFlow >=2.11 for the GroupNormalization layer.
import tensorflow as tf
import keras
from keras import layers, metrics, activations
from RFtune import calMetrics
from geneTokens import tokens2seqs
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
seed=7
loss_name = "MSE"
# loss_name = "BCE"
"""
ddpm:
https://keras.io/examples/generative/ddpm/
classifier guidance:
https://blog.csdn.net/weixin_44966641/article/details/134973550
classifier free guidance:
https://blog.csdn.net/weixin_44966641/article/details/134973550
refer:
https://zhuanlan.zhihu.com/p/588808879

whole token and label trained the previously test datset's result
Generation now is time step：  0
predicted metrics: 
    Accuracy       AUC    Recall     Prec.        F1     Kappa       MCC
0  0.812914  0.890291  0.804388  0.814996  0.809657  0.625734  0.625786
Generation now is time step：  random 
predicted metrics: 
    Accuracy       AUC    Recall     Prec.        F1     Kappa       MCC
0  0.604121  0.656508  0.486798  0.629025  0.548847  0.206257  0.211647
"""
a = np.arange(20) * 500 + 500

latent_node = 25
# train_emb = read_pkl("/home/yanjielu/acp-design/Gene-EC/result/token_train_encoder%dnodes_clip-2.pkl" % latent_node)
# tokens = pd.read_csv("AMP_data/token/amp_cai_5_30.csv")
# non_tokens = pd.read_csv("AMP_data/token/amp_cai_random_5_30.csv")
tokens = pd.read_csv('/home/yanjielu/AMP-sets-20250417/token/amps_2025_token.csv')
non_tokens = pd.read_csv('/home/yanjielu/AMP-sets-20250417/token/amps_2025_rnd_token.csv')

# tokens = pd.read_csv("AMP_data/token/amp_cai_5_30.csv")
# non_tokens = pd.read_csv("AMP_data/token/amp_cai_random_5_30.csv")
# tokens = pd.read_csv("/home/yanjielu/AMP-Gene/AMP_data/token/PeptideAtlas_5_30_del_ampCai_random_token.csv")

all_label = []
label = len(tokens) * [1]
non_label = len(non_tokens) * [0]
tokens_all = pd.concat([tokens, non_tokens])
# tokens_all = tokens.copy()
all_label.extend(label)
all_label.extend(non_label)

tokens_all["label"] = all_label
tokens_all = tokens_all.sample(frac=1, random_state=seed)
label = tokens_all["label"].to_numpy() #.tolist()
token = tokens_all.drop(["label"], axis=1)#.to_numpy() #.tolist()
# data_tokens = tf.data.Dataset.from_tensor_slices({"token": token, "label": label})
# data_tokens = tf.data.Dataset.from_tensor_slices( (token, label) )

batch_size = 5000
# batch_size = 5120
# num_epochs = 7500  # Just for the sake of demonstration
num_epochs = 3000  # Just for the sake of demonstration
total_timesteps = 700
norm_groups = 8  # Number of groups used in GroupNormalization layer
learning_rate = 2e-4

img_size = 32
img_channels = 1
clip_min = -1.0
clip_max = 1.0

first_conv_channels = 16
# channel_multiplier = [1, 2]#, 4, 8]
channel_multiplier = [1]#, 4, 8]
widths = [first_conv_channels * mult for mult in channel_multiplier]
# has_attention = [False, False]#, True, True]
# has_attention = [False, True]#, True, True]
has_attention = [True]#, True, True]
num_res_blocks = 1  # Number of residual blocks


def newest(path):
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    return max(paths, key=os.path.getctime)


def resize_and_rescale(img, size):
    """Resize the image to the desired size first and then
    rescale the pixel values in the range [-1.0, 1.0].

    Args:
        img: Image tensor
        size: Desired image size for resizing
    Returns:
        Resized and rescaled image tensor
    """
    # Resize
    if not tf.is_tensor(img):
        img = tf.cast(img, dtype=tf.float32)
    # img = tf.image.resize(img, size=size, antialias=True)
    img = tf.reshape(img, size)
    img = tf.image.resize_with_pad(img, 32, 1)
    # img = tf.reshape(img, size)

    # Rescale the pixel values
    img = img / 12.5 - 1.0
    img = tf.clip_by_value(img, clip_min, clip_max)
    return img


def train_preprocessing(x):
    img = x.to_numpy()
    img = resize_and_rescale(img, size=(img_size - 1, 1, 1))
    # img = augment(img)
    return img



class GaussianDiffusion:
    """Gaussian diffusion utility.

    Args:
        beta_start: Start value of the scheduled variance
        beta_end: End value of the scheduled variance
        timesteps: Number of time steps in the forward process
    """

    def __init__(
        self,
        beta_start=1e-4,
        beta_end=0.02,
        timesteps=1000,
        clip_min=-1.0,
        clip_max=1.0,
    ):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.timesteps = timesteps
        self.clip_min = clip_min
        self.clip_max = clip_max

        # Define the linear variance schedule
        self.betas = betas = np.linspace(
            beta_start,
            beta_end,
            timesteps,
            dtype=np.float64,  # Using float64 for better precision
        )
        self.num_timesteps = int(timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        self.betas = tf.constant(betas, dtype=tf.float32)
        self.alphas_cumprod = tf.constant(alphas_cumprod, dtype=tf.float32)
        self.alphas_cumprod_prev = tf.constant(alphas_cumprod_prev, dtype=tf.float32)

        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = tf.constant(
            np.sqrt(alphas_cumprod), dtype=tf.float32
        )

        self.sqrt_one_minus_alphas_cumprod = tf.constant(
            np.sqrt(1.0 - alphas_cumprod), dtype=tf.float32
        )

        self.log_one_minus_alphas_cumprod = tf.constant(
            np.log(1.0 - alphas_cumprod), dtype=tf.float32
        )

        self.sqrt_recip_alphas_cumprod = tf.constant(
            np.sqrt(1.0 / alphas_cumprod), dtype=tf.float32
        )
        self.sqrt_recipm1_alphas_cumprod = tf.constant(
            np.sqrt(1.0 / alphas_cumprod - 1), dtype=tf.float32
        )

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_variance = tf.constant(posterior_variance, dtype=tf.float32)

        # Log calculation clipped because the posterior variance is 0 at the beginning
        # of the diffusion chain
        self.posterior_log_variance_clipped = tf.constant(
            np.log(np.maximum(posterior_variance, 1e-20)), dtype=tf.float32
        )

        self.posterior_mean_coef1 = tf.constant(
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
            dtype=tf.float32,
        )

        self.posterior_mean_coef2 = tf.constant(
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
            dtype=tf.float32,
        )

    def _extract(self, a, t, x_shape):
        """Extract some coefficients at specified timesteps,
        then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.

        Args:
            a: Tensor to extract from
            t: Timestep for which the coefficients are to be extracted
            x_shape: Shape of the current batched samples
        """
        batch_size = x_shape[0]
        out = tf.gather(a, t)
        return tf.reshape(out, [batch_size, 1, 1, 1])

    def q_mean_variance(self, x_start, t):
        """Extracts the mean, and the variance at current timestep.

        Args:
            x_start: Initial sample (before the first diffusion step)
            t: Current timestep
        """
        x_start_shape = tf.shape(x_start)
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start_shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start_shape)
        log_variance = self._extract(
            self.log_one_minus_alphas_cumprod, t, x_start_shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise):
        """Diffuse the data.

        Args:
            x_start: Initial sample (before the first diffusion step)
            t: Current timestep
            noise: Gaussian noise to be added at the current timestep
        Returns:
            Diffused samples at timestep `t`
        """
        x_start_shape = tf.shape(x_start)
        return (
            self._extract(self.sqrt_alphas_cumprod, t, tf.shape(x_start)) * x_start
            + self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start_shape)
            * noise
        )

    def predict_start_from_noise(self, x_t, t, noise):
        x_t_shape = tf.shape(x_t)
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t_shape) * x_t
            - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t_shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        """Compute the mean and variance of the diffusion
        posterior q(x_{t-1} | x_t, x_0).

        Args:
            x_start: Stating point(sample) for the posterior computation
            x_t: Sample at timestep `t`
            t: Current timestep
        Returns:
            Posterior mean and variance at current timestep
        """

        x_t_shape = tf.shape(x_t)
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t_shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t_shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t_shape)
        posterior_log_variance_clipped = self._extract(
            self.posterior_log_variance_clipped, t, x_t_shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, pred_noise, x, t, clip_denoised=True):
        x_recon = self.predict_start_from_noise(x, t=t, noise=pred_noise)
        if clip_denoised:
            x_recon = tf.clip_by_value(x_recon, self.clip_min, self.clip_max)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, pred_noise, x, t, clip_denoised=True):
        """Sample from the diffusion model.

        Args:
            pred_noise: Noise predicted by the diffusion model
            x: Samples at a given timestep for which the noise was predicted
            t: Current timestep
            clip_denoised (bool): Whether to clip the predicted noise
                within the specified range or not.
        """
        model_mean, _, model_log_variance = self.p_mean_variance(
            pred_noise, x=x, t=t, clip_denoised=clip_denoised
        )
        noise = tf.random.normal(shape=x.shape, dtype=x.dtype)
        # No noise when t == 0
        nonzero_mask = tf.reshape(
            1 - tf.cast(tf.equal(t, 0), tf.float32), [tf.shape(x)[0], 1, 1, 1]
        )
        return model_mean + nonzero_mask * tf.exp(0.5 * model_log_variance) * noise


# Kernel initializer to use
def kernel_init(scale):
    scale = max(scale, 1e-10)
    return keras.initializers.VarianceScaling(
        scale, mode="fan_avg", distribution="uniform"
    )


class AttentionBlock(layers.Layer):
    """Applies self-attention.

    Args:
        units: Number of units in the dense layers
        groups: Number of groups to be used for GroupNormalization layer
    """

    def __init__(self, units, groups=8, **kwargs):
        self.units = units
        self.groups = groups
        super().__init__(**kwargs)

        self.norm = layers.GroupNormalization(groups=groups)
        self.query = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.key = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.value = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.proj = layers.Dense(units, kernel_initializer=kernel_init(0.0))

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        scale = tf.cast(self.units, tf.float32) ** (-0.5)

        inputs = self.norm(inputs)
        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)

        attn_score = tf.einsum("bhwc, bHWc->bhwHW", q, k) * scale
        attn_score = tf.reshape(attn_score, [batch_size, height, width, height * width])

        attn_score = tf.nn.softmax(attn_score, -1)
        attn_score = tf.reshape(attn_score, [batch_size, height, width, height, width])

        proj = tf.einsum("bhwHW,bHWc->bhwc", attn_score, v)
        proj = self.proj(proj)
        return inputs + proj


class TimeEmbedding(layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.half_dim = dim // 2
        self.emb = math.log(10000) / (self.half_dim - 1)
        self.emb = tf.exp(tf.range(self.half_dim, dtype=tf.float32) * -self.emb)

    def call(self, inputs):
        inputs = tf.cast(inputs, dtype=tf.float32)
        emb = inputs[:, None] * self.emb[None, :]
        emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)
        return emb


class ClassConditioning(layers.Layer):
    def __init__(self, res, num_channels=1):
        super().__init__()
        self.block = keras.Sequential([
            layers.Dense(res * res * num_channels),
            activations.swish(),
            layers.Reshape((res, res, num_channels))
        ])

        self.block.compile()

    def call(self, x):
        return self.block(x)

def ResidualBlock(width, groups=8, activation_fn=activations.swish):
    def apply(inputs):
        x, t = inputs
        input_width = x.shape[3]

        if input_width == width:
            residual = x
        else:
            residual = layers.Conv2D(
                width, kernel_size=1, kernel_initializer=kernel_init(1.0)
            )(x)

        temb = activation_fn(t)
        temb = layers.Dense(width, kernel_initializer=kernel_init(1.0))(temb)[
            :, None, None, :
        ]

        x = layers.GroupNormalization(groups=groups)(x)
        x = activation_fn(x)
        x = layers.Conv2D(
            width, kernel_size=(3,1), padding="same", kernel_initializer=kernel_init(1.0)
        )(x)

        x = layers.Add()([x, temb])
        x = layers.GroupNormalization(groups=groups)(x)
        x = activation_fn(x)

        x = layers.Conv2D(
            width, kernel_size=(3,1), padding="same", kernel_initializer=kernel_init(0.0)
        )(x)
        x = layers.Add()([x, residual])
        return x

    return apply


def DownSample(width):
    def apply(x):
        x = layers.Conv2D(
            width,
            kernel_size=(3,1),
            strides=(2,1),
            padding="same",
            kernel_initializer=kernel_init(1.0),
        )(x)
        return x

    return apply


def UpSample(width, interpolation="nearest"):
    def apply(x):
        x = layers.UpSampling2D(size=(2,1), interpolation=interpolation)(x)
        x = layers.Conv2D(
            width, kernel_size=(3,1), padding="same", kernel_initializer=kernel_init(1.0)
        )(x)
        return x

    return apply


def TimeMLP(units, activation_fn=activations.swish):
    def apply(inputs):
        temb = layers.Dense(
            units, activation=activation_fn, kernel_initializer=kernel_init(1.0)
        )(inputs)
        temb = layers.Dense(units, kernel_initializer=kernel_init(1.0))(temb)
        return temb

    return apply


def build_model(
    img_size,
    img_channels,
    widths,
    has_attention,
    num_res_blocks=2,
    norm_groups=8,
    interpolation="nearest",
    activation_fn=activations.swish,
):
    image_input = layers.Input(
        shape=(img_size, 1, img_channels), name="image_input"
    )
    time_input = keras.Input(shape=(), dtype=tf.int64, name="time_input")

    x = layers.Conv2D(
        first_conv_channels,
        kernel_size=(3, 1),
        padding="same",
        kernel_initializer=kernel_init(1.0),
    )(image_input)

    temb = TimeEmbedding(dim=first_conv_channels * 4)(time_input)
    temb = TimeMLP(units=first_conv_channels * 4, activation_fn=activation_fn)(temb)

    skips = [x]

    # DownBlock
    for i in range(len(widths)):
        for _ in range(num_res_blocks):
            x = ResidualBlock(
                widths[i], groups=norm_groups, activation_fn=activation_fn
            )([x, temb])
            if has_attention[i]:
                x = AttentionBlock(widths[i], groups=norm_groups)(x)
            skips.append(x)

        if widths[i] != widths[-1]:
            x = DownSample(widths[i])(x)
            skips.append(x)

    # MiddleBlock
    x = ResidualBlock(widths[-1], groups=norm_groups, activation_fn=activation_fn)(
        [x, temb]
    )
    x = AttentionBlock(widths[-1], groups=norm_groups)(x)
    x = ResidualBlock(widths[-1], groups=norm_groups, activation_fn=activation_fn)(
        [x, temb]
    )

    # UpBlock
    for i in reversed(range(len(widths))):
        for _ in range(num_res_blocks + 1):
            x = layers.Concatenate(axis=-1)([x, skips.pop()])
            x = ResidualBlock(
                widths[i], groups=norm_groups, activation_fn=activation_fn
            )([x, temb])
            if has_attention[i]:
                x = AttentionBlock(widths[i], groups=norm_groups)(x)

        if i != 0:
            x = UpSample(widths[i], interpolation=interpolation)(x)

    # End block
    x = layers.GroupNormalization(groups=norm_groups)(x)
    x = activation_fn(x)
    x = layers.Conv2D(1, (3, 1), padding="same", kernel_initializer=kernel_init(0.0))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1)(x)
    x = activations.sigmoid(x)
    return keras.Model([image_input, time_input], x, name="unet")

class DiffusionModel(keras.Model):
    def __init__(self, network, timesteps, gdf_util, clip_int=False, clip=False):
        super().__init__()
        self.network = network
        self.timesteps = timesteps
        self.gdf_util = gdf_util
        self.clip_int = clip_int
        self.clip = clip
    def calculateLable(self, y, t):
        ratio = (t + 1) / self.timesteps
        ratio = tf.cast(tf.reshape(ratio, tf.shape(y)), dtype=y.dtype)

        # ratio = tf.where(t < 599, tf.zeros_like(t), t)

        y_ratio = tf.math.multiply(ratio, y)
        # y_ratio = y
        return y_ratio

    def train_step(self, data):
        # 1. Get the batch sizeori
        images, y = data
        batch_size = tf.shape(images)[0]

        # 2. Sample timesteps uniformly
        t = tf.random.uniform(
            minval=0, maxval=self.timesteps, shape=(batch_size,), dtype=tf.int64
        )

        # y_ratio = self.calculateLable(y, t)
        y_ratio = y

        with tf.GradientTape() as tape:
            # 3. Sample random noise to be added to the images in the batch
            noise = tf.random.normal(shape=tf.shape(images), dtype=images.dtype)

            # 4. Diffuse the images with noise
            images_t = self.gdf_util.q_sample(images, t, noise)

            # clip to the tokens that can be used to converted to Seqences
            if self.clip_int:
                images_t = tf.round(images_t * 12.5 + 12.5)
                images_t = tf.clip_by_value(images_t, 0.0, 25.0)
                images_t = images_t / 12.5 - 1
            elif self.clip:
                images_t = tf.clip_by_value(images_t, clip_min, clip_max)

            # 5. Pass the diffused images and time steps to the network
            pred_y = self.network([images_t, t], training=True)

            loss = self.loss(y_ratio, pred_y)


        # 7. Get the gradients
        gradients = tape.gradient(loss, self.network.trainable_weights)

        # 8. Update the weights of the network
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        return {"loss": loss}

    def predict_images_t(self, images, t=0, random_t=False):

        num_images = tf.shape(images)[0]
        noise = tf.random.normal(shape=tf.shape(images), dtype=images.dtype)
        if random_t:
            tt = tf.random.uniform(
                minval=0, maxval=self.timesteps, shape=(num_images,), dtype=tf.int64
            )
        else:
            tt = tf.cast(tf.fill(num_images, t), dtype=tf.int64)
        # tt = tf.cast(tf.fill(num_images, t), dtype=tf.int64)
        images_t = self.gdf_util.q_sample(images, tt, noise)
        if self.clip_int:
            images_t = tf.round(images_t * 12.5 + 12.5)
            images_t = tf.clip_by_value(images_t, 0.0, 25.0)
            images_t = images_t / 12.5 - 1
        elif self.clip:
            images_t = tf.clip_by_value(images_t, clip_min, clip_max)
        y_pre = self.network.predict(
            [images_t, tt], verbose=0, batch_size=batch_size
        )
        return y_pre

    def train_test_t(self, X_test, y_test, t=0, random_t=False):
        images = tf.cast(X_test, dtype=tf.float32)
        noise = tf.random.normal(shape=tf.shape(images), dtype=images.dtype)
        num_images = tf.shape(X_test)[0]
        y = tf.cast(tf.reshape(y_test, (len(y_test), 1)), dtype=tf.int64)

        if random_t:
            tt = tf.random.uniform(
                minval=0, maxval=self.timesteps, shape=(num_images,), dtype=tf.int64
            )
        else:
            tt = tf.cast(tf.fill(num_images, t), dtype=tf.int64)
        images_t = self.gdf_util.q_sample(images, tt, noise)
        # images_t = tf.Variable(images_t, trainable=True)
        # y = tf.Variable(y, trainable=True)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(images_t)
            if self.clip_int:
                images_t = tf.round(images_t * 12.5 + 12.5)
                images_t = tf.clip_by_value(images_t, 0.0, 25.0)
                images_t = images_t / 12.5 - 1
            elif self.clip:
                images_t = tf.clip_by_value(images_t, clip_min, clip_max)
            pred_y = self.network([images_t, tt], training=False)
            loss = self.loss(y, pred_y)
        # del tape
        grad_pred = tape.gradient(pred_y, images_t)
        grad_loss = tape.gradient(loss, images_t)
        # gradients = tape.gradient(loss, self.network.trainable_weights)
        return grad_pred, grad_loss, images_t, y, pred_y

    def calMet(self, y_ori, y_pre):
        if tf.is_tensor(y_pre):
            y_prob = y_pre.numpy()
        else:
            y_prob = y_pre.copy()
        y_true = np.reshape(y_ori, y_pre.shape)
        y_cls = y_prob.copy()
        y_cls[y_cls >= 0.5] = 1
        y_cls[y_cls < 0.5] = 0
        met = calMetrics(y_true, y_cls, y_prob)
        print("predicted metrics: \n", met)
        return met


# Build the unet model
network = build_model(
    img_size=img_size,
    img_channels=img_channels,
    widths=widths,
    has_attention=has_attention,
    num_res_blocks=num_res_blocks,
    norm_groups=norm_groups,
    activation_fn=activations.swish,
)


# Get an instance of the Gaussian Diffusion utilities
gdf_util = GaussianDiffusion(timesteps=total_timesteps)

# Get the model
clsMdl = DiffusionModel(
    network=network,
    # ema_network=ema_network,
    gdf_util=gdf_util,
    timesteps=total_timesteps,
)
# import tensorflow_addons as tfa
# Compile the model
if loss_name == "BCE":
    clsMdl.compile(
        loss=keras.losses.BinaryCrossentropy(from_logits=False),
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    )
if loss_name == "MSE":
    clsMdl.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    )

# checkpoint_path = "checkpoints/diffusion_model_pretrain"
# checkpoint_path = "checkpoints/diffusion_model_train"
# folder = ("checkpoints/train-classifer/%depoch_%dtiemstep_%s_clip" %
#           (num_epochs, total_timesteps, loss_name))
# folder = ("checkpoints/train-classifer/%depoch_%dtiemstep" %
#           (num_epochs, total_timesteps))
folder = ("checkpoints/train-classifer/2025amps/%depoch_%dtiemstep" %
          (num_epochs, total_timesteps))
checkpoint_path = "%s/whole_classifier_sigmoid_Unet_model_train" % (folder)

checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    monitor="loss",
    mode="min",
    save_best_only=True,
)

train_fts = token.apply(train_preprocessing, axis=1)
tokens_all = np.array(list(map(np.array, list(train_fts))))
labels_all = label

# history = clsMdl.fit(
#     # data_tokens,
#     tokens_all,
#     labels_all,
#     # validation_data=(X_test,y_test),
#     epochs=num_epochs,
#     batch_size=batch_size,
#     callbacks=[
#         checkpoint_callback,
#     ],
# )

# load the best model and generate AMPs
ps = pd.read_csv(folder + "/" + "checkpoint", delimiter=" ", header=None)
p1 = folder + "/" + ps[1][0]
clsMdl.load_weights(p1)

if __name__ == "__main__":
    # grad_pred, grad_loss, imgs_t, y_true, y_pred = model.train_test_t(X_test, y_test, random_t=True)
    # grad = model.train_test_t(X_test, y_test, t=499, random_t=False)
    # data_tokens = tf.data.Dataset.from_tensor_slices((train_tokens,train_labels))
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(tokens_all, labels_all, test_size=0.1, random_state=seed)

    # predict all time steps
    mets = []
    for t in np.arange(0, total_timesteps):
        y_pre = clsMdl.predict_images_t(X_test, t=t)
        met = clsMdl.calMet(y_test, y_pre)
        mets.append(met)
    mets = pd.concat(mets)
    mets.index = np.arange(0, total_timesteps)

    mets.plot()
    plt.xlabel("Time Step")
    plt.ylabel("Score")
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.axhline(y = 0.5, color = 'black', linestyle = '-.', linewidth=0.5)
    plt.savefig("./pics/t_Noised_AMP_Classifier-different-timesteps-Yan-Model_%depoch_%dtiemstep_%s-metrics.png" %
                (num_epochs, total_timesteps, loss_name))

    plt.close()

#
# y_pre_rnd = model.predict_images_t(X_test, random_t=True)
# met_rnd = model.calMet(y_test, y_pre_rnd)

# inputs = np.concatenate((input_train, input_test), axis=0)
# targets = np.concatenate((target_train, target_test), axis=0)
#
# # Define the K-fold Cross Validator
# kfold = KFold(n_splits=num_folds, shuffle=True)
#
# # K-fold Cross Validation model evaluation
# fold_no = 1
# for train, test in kfold.split(inputs, targets):
