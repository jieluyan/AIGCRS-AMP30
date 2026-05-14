import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Requires TensorFlow >=2.11 for the GroupNormalization layer.
import tensorflow as tf
import keras
from keras import layers
from geneTokens import tokens2seqs
import os
from RFtune import importFts, used_fts
from tools.base import seqs2fastas
import joblib
from RFtune import calMetrics
from ClassifierUnetLoadMdl import clsMdl, loss_name
"""
ddpm:
https://keras.io/examples/generative/ddpm/
classifier guidance:
https://blog.csdn.net/weixin_44966641/article/details/134973550
classifier free guidance:
https://blog.csdn.net/weixin_44966641/article/details/134973550
refer:
https://zhuanlan.zhihu.com/p/588808879
"""
# for total_timesteps in [7000, 8000, 9000, 10000]:
a = np.arange(20) * 500 + 500
guidance_scale = 7.5
guidance_scale = 0.75
def RFDevelopAndTest(test_path, rf_mdl="/home/yanjielu/AMP-Gene/model/random_forest_ntree2000_log2.joblib"):
    fts = importFts(test_path, ft_list=used_fts, class_val=None)
    rf = joblib.load(rf_mdl)
    y_pre = rf.predict(fts)
    prob = rf.predict_proba(fts)
    return y_pre, prob[:, 1]

latent_node = 25
# train_emb = read_pkl("/home/yanjielu/acp-design/Gene-EC/result/token_train_encoder%dnodes_clip-2.pkl" % latent_node)
tokens = pd.read_csv("/home/yanjielu/AMP-Gene/AMP_data/token/amp_cai_5_30.csv")
# tokens = pd.read_csv("/home/yanjielu/AMP-Gene/AMP_data/token/PeptideAtlas_5_30_del_ampCai_random_token.csv")
# latent_label, latent_token = train_emb["label_embeddings"], train_emb["token_embeddings"]
latent_token = tokens.to_numpy().reshape((-1, 31, 1))
input_channels = latent_token.shape[-1]
image_size = latent_token.shape[1]
in_shape = latent_token.shape[1:]

batch_size = 5120
num_epochs = 5000#4500  # Just for the sake of demonstration
total_timesteps = 700#200
norm_groups = 8  # Number of groups used in GroupNormalization layer
learning_rate = 2e-4

img_size = 32
img_channels = 1
clip_min = -1.0
clip_max = 1.0

first_conv_channels = 64
channel_multiplier = [1, 2, 4, 8]
widths = [first_conv_channels * mult for mult in channel_multiplier]
has_attention = [False, False, True, True]
num_res_blocks = 2  # Number of residual blocks

# dataset_name = "oxford_flowers102"
# splits = ["train"]
#
# (ds,) = tfds.load(dataset_name, split=splits, with_info=False, shuffle_files=True)
from sklearn.metrics import confusion_matrix
from pycaret.utils.generic import check_metric
from ifeature.codes import *

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

    # height = tf.shape(img)[0]
    # width = tf.shape(img)[1]
    # crop_size = tf.minimum(height, width)
    #
    # img = tf.image.crop_to_bounding_box(
    #     img,
    #     (height - crop_size) // 2,
    #     (width - crop_size) // 2,
    #     crop_size,
    #     crop_size,
    # )

    # Resize
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
    img = x
    img = resize_and_rescale(img, size=(img_size - 1, 1, 1))
    # img = augment(img)
    return img


data_tokens = tf.data.Dataset.from_tensor_slices((tokens))

train_ds = (
    data_tokens.map(train_preprocessing, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size, drop_remainder=True)
    .shuffle(batch_size * 2)
    .prefetch(tf.data.AUTOTUNE)
)


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
            keras.activations.swish(),
            layers.Reshape((res, res, num_channels))
        ])

        self.block.compile()

    def call(self, x):
        return self.block(x)

def ResidualBlock(width, groups=8, activation_fn=keras.activations.swish):
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


def TimeMLP(units, activation_fn=keras.activations.swish):
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
    activation_fn=keras.activations.swish,
):
    image_input = layers.Input(
        shape=(img_size, 1, img_channels), name="image_input"
    )
    time_input = keras.Input(shape=(), dtype=tf.int64, name="time_input")
    # num_classes = 1,
    # class_embedder = None,
    # class_emb_dim = 64,
    # class_embeddings = layers.Embedding(num_classes, class_emb_dim) if class_embedder is None else class_embedder

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
    return keras.Model([image_input, time_input], x, name="unet")

class DiffusionModel(keras.Model):
    def __init__(self, network, ema_network, timesteps, gdf_util, ema=0.999, clip_int=False, clip=True):
        super().__init__()
        self.network = network
        self.ema_network = ema_network
        self.timesteps = timesteps
        self.gdf_util = gdf_util
        self.ema = ema
        self.accs = []
        self.clip_int = clip_int
        self.clip = clip

    def train_step(self, images):
        # 1. Get the batch size
        batch_size = tf.shape(images)[0]

        # 2. Sample timesteps uniformly
        t = tf.random.uniform(
            minval=0, maxval=self.timesteps, shape=(batch_size,), dtype=tf.int64
        )

        with tf.GradientTape() as tape:
            # 3. Sample random noise to be added to the images in the batch
            noise = tf.random.normal(shape=tf.shape(images), dtype=images.dtype)

            # 4. Diffuse the images with noise
            images_t = self.gdf_util.q_sample(images, t, noise)
            if self.clip_int:
                images_t = tf.round(images_t * 12.5 + 12.5)
                images_t = tf.clip_by_value(images_t, 0.0, 25.0)
                images_t = images_t / 12.5 - 1
            elif self.clip:
                images_t = tf.clip_by_value(images_t, clip_min, clip_max)
            # 5. Pass the diffused images and time steps to the network
            pred_noise = self.network([images_t, t], training=True)

            loss = self.loss(noise, pred_noise)

        # 7. Get the gradients
        gradients = tape.gradient(loss, self.network.trainable_weights)

        # 8. Update the weights of the network
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        # 9. Updates the weight values for the network with EMA weights
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        # 10. Return loss values
        return {"loss": loss}

    def generate_images(self, num_images=16):
        # 1. Randomly sample noise (starting point for reverse process)
        samples = tf.random.normal(
            shape=(num_images, img_size, 1, img_channels), dtype=tf.float32
        )
        # 2. Sample from the model iteratively
        for t in reversed(range(0, self.timesteps)):
            tt = tf.cast(tf.fill(num_images, t), dtype=tf.int64)
            cls_loss = self.sample2seqs(samples, t)
            # print("time step: ", t)
            pred_noise = self.ema_network.predict(
                [samples, tt], verbose=0, batch_size=num_images
            )
            samples = self.gdf_util.p_sample(
                pred_noise, samples, tt, clip_denoised=True
            )
            if self.clip_int:
                samples = tf.round(samples * 12.5 + 12.5)
                samples = tf.clip_by_value(samples, 0.0, 25.0)
                samples = samples / 12.5 - 1
            elif self.clip:
                samples = tf.clip_by_value(samples, clip_min, clip_max)
        # 3. Return generated samples
        return samples

    def controlSeqsLen(self, seqs, max_len=30, min_len=5):
        new_seqs = []
        for seq in seqs:
            if len(seq) >= min_len and len(seq) <= max_len:
                new_seqs.append(seq)
        return new_seqs
    def sample2seqs(self, samples, t, c=7.5, target=1):
        generated_samples = (
            tf.clip_by_value(samples * 12.5 + 12.5, 0.0, 25.0)
            .numpy()
            .astype(np.uint8)
        )
        gene_tokens = generated_samples.squeeze()
        seqs = tokens2seqs(gene_tokens, scaled=False)
        label = tf.cast(tf.fill(len(seqs), target), dtype=tf.float32)
        new_seqs = self.controlSeqsLen(seqs)
        l0 = len(seqs) - len(new_seqs)
        reverse_target = 1-target
        added = tf.cast(tf.fill(l0, reverse_target), dtype=tf.float32)
        if len(new_seqs) == 0:
            new_prob = added
        else:
            test_path = "result/cls_timestep/temp_timestep%d.fastas" % t
            df = seqs2fastas(new_seqs, path=test_path)
            cls, prob = RFDevelopAndTest(test_path)
            size = prob.shape
            l = size[0]
            acc = sum(cls)/l
            prob = tf.cast(prob, dtype=tf.float32)
            new_prob = tf.concat([prob, added], 0)

        # tf.log
        cls_loss = self.loss(new_prob, label) * c
        return cls_loss

    def genetoken2seqs(self, generated_tokens):
        generated_samples = (
            tf.clip_by_value(generated_tokens * 12.5 + 12.5, 0.0, 25.0)
            .numpy()
            .astype(np.uint8)
        )
        gene_tokens = generated_samples.squeeze()
        seqs = tokens2seqs(gene_tokens, scaled=False)
        return seqs
    def generate_AMPs(self, num=10000, label=1):
        # plot random generated images for visual evaluation of generation quality
        y_label = tf.cast(tf.fill(num, label), dtype=tf.int64)
        generated_tokens, best_tokens, mets, best_acc = self.geneTokens_refineByCls(num_images=num, y_label=y_label)
        seqs = self.genetoken2seqs(generated_tokens)
        best_seqs = self.genetoken2seqs(best_tokens)
        return seqs, best_seqs, mets, best_acc

    def geneTokens_refineByCls(self, num_images, y_label):
        # images = tf.cast(X_test, dtype=tf.float32)
        # noise = tf.random.normal(shape=tf.shape(images), dtype=images.dtype)
        samples = tf.random.normal(
            shape=(num_images, img_size, 1, img_channels), dtype=tf.float32
        )
        y = tf.cast(tf.reshape(y_label, (len(y_label), 1)), dtype=tf.int64)
        # 2. Sample from the model iteratively
        mets = []
        best_token = samples
        best_acc = 0
        for t in reversed(range(0, self.timesteps)):
            tt = tf.cast(tf.fill(num_images, t), dtype=tf.int64)
            pred_noise = self.ema_network.predict(
                [samples, tt], verbose=0, batch_size=5000
            )
            # if t % 100  == 0:
            #     print("time")
            # pred_noise = tf.stop_gradient(pred_noise)
            # pred_noise = pred_noise.numpy()
            pred_noise = tf.cast(pred_noise, dtype=samples.dtype)
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(samples)
                if self.clip_int:
                    samples = tf.round(samples * 12.5 + 12.5)
                    samples = tf.clip_by_value(samples, 0.0, 25.0)
                    samples = samples / 12.5 - 1
                elif self.clip:
                    samples = tf.clip_by_value(samples, clip_min, clip_max)
                pred_y = clsMdl.network([samples, tt], training=False)
                loss = clsMdl.loss(y, pred_y)
            # del tape
            # grad_pred = tape.gradient(pred_y, samples)
            grad_loss = tape.gradient(loss, samples)
            # samples += grad_pred * guidance_scale
            # scale = tf.cast(tf.fill(num_images, guidance_scale), dtype=grad_loss.dtype)
            pred_noise += guidance_scale * grad_loss
            met = clsMdl.calMet(y.numpy(), pred_y.numpy())
            if met["Accuracy"][0] > best_acc:
                best_token = samples
                best_acc = met["Accuracy"][0]
            samples = self.gdf_util.p_sample(
                pred_noise, samples, tt, clip_denoised=True
            )
            mets.append(met)
        mets = pd.concat(mets)
        p_mets = ("./pics/rf_changed/newClassifierGuidance-Diffusion-Yan-Model_%depoch_%dtiemstep_%s_%.2fscale_clip-metrics600cls.csv" %
                    (num_epochs, total_timesteps, loss_name, guidance_scale))
        mets.to_csv(p_mets, header=True, index=False)
        mets.index = reversed(range(0, self.timesteps))
        mets[["Accuracy", "F1"]].plot()
        plt.xlabel("Time Step")
        plt.ylabel("Score")
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.axhline(y=0.5, color='black', linestyle='-.', linewidth=0.5)
        plt.gca().invert_xaxis()
        plt.savefig("./pics/rf_changed/newClassifierGuidance-Diffusion-Yan-Model_%depoch_%dtiemstep_%s_%.2fscale_clip-metrics600cls.png" %
                    (num_epochs, total_timesteps, loss_name, guidance_scale))
        plt.close()
        return samples, best_token, mets, best_acc

    def calMet(self, y_ori, y_pre):
        if tf.is_tensor(y_pre):
            y_prob = y_pre.numpy()
        else:
            y_prob = y_pre.copy()

        if tf.is_tensor(y_ori):
            y_ori = y_ori.numpy()

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
    activation_fn=keras.activations.swish,
)
ema_network = build_model(
    img_size=img_size,
    img_channels=img_channels,
    widths=widths,
    has_attention=has_attention,
    num_res_blocks=num_res_blocks,
    norm_groups=norm_groups,
    activation_fn=keras.activations.swish,
)
ema_network.set_weights(network.get_weights())  # Initially the weights are the same

# Get an instance of the Gaussian Diffusion utilities
gdf_util = GaussianDiffusion(timesteps=total_timesteps)

# Get the model
model = DiffusionModel(
    network=network,
    ema_network=ema_network,
    gdf_util=gdf_util,
    timesteps=total_timesteps,
)

# Compile the model
model.compile(
    loss=keras.losses.MeanSquaredError(),
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
) 

# checkpoint_path = "checkpoints/diffusion_model_pretrain"
# checkpoint_path = "checkpoints/diffusion_model_train"
folder = "checkpoints/train/%depoch_%dtiemstep_clip" % (num_epochs, total_timesteps)
folder = "checkpoints/train/rf_changed_%depoch_%dtiemstep_MSE_Gaussian" % (num_epochs, total_timesteps)
checkpoint_path = "%s/diffusion_model_train" % (folder)

ps = pd.read_csv(folder + "/" + "checkpoint", delimiter=" ", header=None)
p1 = folder + "/" + ps[1][0]
model.load_weights(p1)
# generated_tokens = model.generate_images(num_images=64)
# predict all time steps
new_seqs = []
generated_seqs, best_seqs, mets, best_acc = model.generate_AMPs(num=10000, label=1)
for seq in generated_seqs:
    if len(seq) > 4 and len(seq) < 31:
        new_seqs.append(seq)
id = ["geneAMP_%d" % (i+1) for i in range(len(new_seqs))]
df = pd.DataFrame({"ID": id, "SEQUENCE": new_seqs})

p_generated_seq = ("~/AMP-Gene-iseLab/result/gene_seqs/Classifier_guidance/rf_changed_ClassifierGuidance-Diffusion_%depoch_%dtiemstep_%.2fscale_clip600cls.csv"
                   % (num_epochs, total_timesteps, guidance_scale))
# p_generated_seq = "/home/yanjielu/AMP-Gene/result/gene_seqs/amp_cai_generated_seqs_5_30_pretrain_epoch1000.csv"
df.to_csv(p_generated_seq, header=True, index=False)
new_best_seqs = []
for seq in best_seqs:
    if len(seq) > 4 and len(seq) < 31:
        new_best_seqs.append(seq)
id = ["geneAMP_%d" % (i+1) for i in range(len(new_best_seqs))]
df_best = pd.DataFrame({"ID": id, "SEQUENCE": new_best_seqs})
p_generated_seq_best = ("~/AMP-Gene-iseLab/result/gene_seqs/Classifier_guidance/rf_changed_ClassifierGuidance-Diffusion_%depoch_%dtiemstep_%.2fscale_clip600cls_best.csv"
                   % (num_epochs, total_timesteps, guidance_scale))
# p_generated_seq = "/home/yanjielu/AMP-Gene/result/gene_seqs/amp_cai_generated_seqs_5_30_pretrain_epoch1000.csv"
df_best.to_csv(p_generated_seq_best, header=True, index=False)

def plotLosses(hist):
    plt.plot(hist["loss"])
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.show()
    plt.savefig("Diffusion-token-Yan-Model_%depoch_%dtiemstep_%.1fscale-losses.png" % (num_epochs, total_timesteps, guidance_scale))
    return

# plotLosses(history)
