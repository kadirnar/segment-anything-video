import gradio as gr

from metaseg import SegAutoMaskGenerator


def image_app():
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                seg_automask_image_file = gr.Image(type="filepath").style(height=260)

                with gr.Row():
                    with gr.Column():
                        seg_automask_image_model_type = gr.Dropdown(
                            choices=[
                                "vit_h",
                                "vit_l",
                                "vit_b",
                            ],
                            value="vit_l",
                            label="Model Type",
                        )

                        seg_automask_image_points_per_side = gr.Slider(
                            minimum=0,
                            maximum=32,
                            step=2,
                            value=16,
                            label="Points per Side",
                        )

                        seg_automask_image_points_per_batch = gr.Slider(
                            minimum=0,
                            maximum=64,
                            step=2,
                            value=64,
                            label="Points per Batch",
                        )

                        seg_automask_image_min_area = gr.Number(
                            value=0,
                            label="Min Area",
                        )

                seg_automask_image_predict = gr.Button(value="Generator")

            with gr.Column():
                output_image = gr.Image()

        seg_automask_image_predict.click(
            fn=SegAutoMaskGenerator().save_image,
            inputs=[
                seg_automask_image_file,
                seg_automask_image_model_type,
                seg_automask_image_points_per_side,
                seg_automask_image_points_per_batch,
                seg_automask_image_min_area,
            ],
            outputs=[output_image],
        )


def video_app():
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                seg_automask_video_file = gr.Video().style(height=260)

                with gr.Row():
                    with gr.Column():
                        seg_automask_video_model_type = gr.Dropdown(
                            choices=[
                                "vit_h",
                                "vit_l",
                                "vit_b",
                            ],
                            value="vit_l",
                            label="Model Type",
                        )

                        seg_automask_video_points_per_side = gr.Slider(
                            minimum=0,
                            maximum=32,
                            step=2,
                            value=16,
                            label="Points per Side",
                        )
                        seg_automask_video_points_per_batch = gr.Slider(
                            minimum=0,
                            maximum=64,
                            step=2,
                            value=64,
                            label="Points per Batch",
                        )
                        with gr.Row():
                            with gr.Column():
                                seg_automask_video_min_area = gr.Number(
                                    value=1000,
                                    label="Min Area",
                                )

                seg_automask_video_predict = gr.Button(value="Generator")
            with gr.Column():
                output_video = gr.Video()

        seg_automask_video_predict.click(
            fn=SegAutoMaskGenerator().save_video,
            inputs=[
                seg_automask_video_file,
                seg_automask_video_model_type,
                seg_automask_video_points_per_side,
                seg_automask_video_points_per_batch,
                seg_automask_video_min_area,
            ],
            outputs=[output_video],
        )


def metaseg_app():
    app = gr.Blocks()
    with app:
        with gr.Row():
            with gr.Column():
                with gr.Tab("Image"):
                    image_app()
                with gr.Tab("Video"):
                    video_app()

    app.queue(concurrency_count=2)
    app.launch(debug=True, enable_queue=True)


if __name__ == "__main__":
    metaseg_app()
