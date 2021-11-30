import os
from PIL import Image
import streamlit as st
from tensorflow.keras import utils, backend

from enhance_me import MIRNet, ZeroDCE


def get_mirnet_object() -> MIRNet:
    utils.get_file(
        "weights_lol_128.h5",
        "https://github.com/soumik12345/enhance-me/releases/download/v0.2/weights_lol_128.h5",
        cache_dir=".",
        cache_subdir="weights",
    )
    mirnet = MIRNet()
    mirnet.build_model()
    mirnet.load_weights("./weights/weights_lol_128.h5")
    return mirnet


def get_zero_dce_object(model_alias: str) -> ZeroDCE:
    utils.get_file(
        f"{model_alias}.h5",
        f"https://github.com/soumik12345/enhance-me/releases/download/v0.4/{model_alias}.h5",
        cache_dir=".",
        cache_subdir="weights",
    )
    dce = ZeroDCE()
    dce.load_weights(os.path.join("./weights", f"{model_alias}.h5"))
    return dce


def main():
    st.markdown("# Enhance Me")
    st.markdown("Made with :heart: by [geekyRakshit](http://github.com/soumik12345)")
    application = st.sidebar.selectbox(
        "Please select the application:", ("", "Low-light enhancement")
    )
    if application != "":
        if application == "Low-light enhancement":
            uploaded_file = st.sidebar.file_uploader("Select your image:")
            if uploaded_file is not None:
                original_image = Image.open(uploaded_file)
                st.image(original_image, caption="original image")
                model_option = st.sidebar.selectbox(
                    "Please select the model:",
                    (
                        "",
                        "MIRNet",
                        "Zero-DCE (dce_weights_lol_128)",
                        "Zero-DCE (dce_weights_lol_128_resize)",
                        "Zero-DCE (dce_weights_lol_256)",
                        "Zero-DCE (dce_weights_lol_256_resize)",
                        "Zero-DCE (dce_weights_unpaired_128)",
                        "Zero-DCE (dce_weights_unpaired_128_resize)",
                        "Zero-DCE (dce_weights_unpaired_256)",
                        "Zero-DCE (dce_weights_unpaired_256_resize)"
                    ),
                )
                if model_option != "":
                    if model_option == "MIRNet":
                        st.sidebar.info("Loading MIRNet...")
                        mirnet = get_mirnet_object()
                        st.sidebar.info("Done!")
                        st.sidebar.info("Processing Image...")
                        enhanced_image = mirnet.infer(original_image)
                        st.sidebar.info("Done!")
                        st.image(enhanced_image, caption="enhanced image")
                    elif "Zero-DCE" in model_option:
                        model_alias = model_option[model_option.find("(") + 1: model_option.find(")")]
                        st.sidebar.info("Loading Zero-DCE...")
                        zero_dce = get_zero_dce_object(model_alias)
                        st.sidebar.info("Done!")
                        enhanced_image = zero_dce.infer(original_image)
                        st.sidebar.info("Done!")
                        st.image(enhanced_image, caption="enhanced image")
                    backend.clear_session()


if __name__ == "__main__":
    main()
