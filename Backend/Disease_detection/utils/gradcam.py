import os

import cv2
import numpy as np
import tensorflow as tf


def generate_gradcam(
    model,
    image_array,
    save_path=None,
    last_conv_layer_name="top_conv",
    base_model_name="efficientnetb0"
):

    # =====================================================
    # GET NESTED EFFICIENTNET MODEL
    # =====================================================

    base_model = model.get_layer(
        base_model_name
    )


    # Target convolution layer inside EfficientNetB0
    target_layer = base_model.get_layer(
        last_conv_layer_name
    )


    # =====================================================
    # MODEL THAT RETURNS:
    # 1. target convolution output
    # 2. EfficientNet final feature output
    # =====================================================

    base_grad_model = tf.keras.models.Model(
        inputs=base_model.inputs,
        outputs=[
            target_layer.output,
            base_model.output
        ]
    )


    # Find where EfficientNet is located
    base_model_index = model.layers.index(
        base_model
    )


    # =====================================================
    # FORWARD PASS + GRADIENT
    # =====================================================

    with tf.GradientTape() as tape:

        x = image_array


        # Run layers before EfficientNet
        for layer in model.layers[
            1:base_model_index
        ]:

            x = layer(
                x,
                training=False
            )


        # Run EfficientNet and capture top_conv
        conv_outputs, x = base_grad_model(
            x,
            training=False
        )


        # Run classification layers after EfficientNet
        for layer in model.layers[
            base_model_index + 1:
        ]:

            try:

                x = layer(
                    x,
                    training=False
                )

            except TypeError:

                x = layer(x)


        predictions = x


        predicted_class = tf.argmax(
            predictions[0]
        )


        class_score = predictions[
            :,
            predicted_class
        ]


    # =====================================================
    # COMPUTE GRADIENTS
    # =====================================================

    gradients = tape.gradient(
        class_score,
        conv_outputs
    )


    if gradients is None:

        raise ValueError(
            "Grad-CAM gradients could not be calculated."
        )


    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )


    conv_outputs = conv_outputs[0]


    heatmap = tf.reduce_sum(
        conv_outputs
        * pooled_gradients,
        axis=-1
    )


    # Keep only positive influence
    heatmap = tf.maximum(
        heatmap,
        0
    )


    max_value = tf.reduce_max(
        heatmap
    )


    heatmap = heatmap / (
        max_value + 1e-8
    )


    heatmap = heatmap.numpy()


    # Resize to input image size
    heatmap = cv2.resize(
        heatmap,
        (224, 224)
    )


    # =====================================================
    # SAVE VISUAL GRAD-CAM OVERLAY
    # =====================================================

    if save_path is not None:

        output_directory = os.path.dirname(
            save_path
        )


        if output_directory:

            os.makedirs(
                output_directory,
                exist_ok=True
            )


        heatmap_uint8 = np.uint8(
            255 * heatmap
        )


        colored_heatmap = cv2.applyColorMap(
            heatmap_uint8,
            cv2.COLORMAP_JET
        )


        colored_heatmap = cv2.cvtColor(
            colored_heatmap,
            cv2.COLOR_BGR2RGB
        )


        original_image = np.clip(
            image_array[0],
            0,
            255
        ).astype(
            np.uint8
        )


        overlay = cv2.addWeighted(
            original_image,
            0.60,
            colored_heatmap,
            0.40,
            0
        )


        overlay_bgr = cv2.cvtColor(
            overlay,
            cv2.COLOR_RGB2BGR
        )


        success = cv2.imwrite(
            save_path,
            overlay_bgr
        )


        if not success:

            raise ValueError(
                "Grad-CAM image could not be saved."
            )


    return heatmap