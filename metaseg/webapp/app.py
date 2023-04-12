import gradio as gr
from demo import automask_image_app, automask_video_app, sahi_autoseg_app


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

                        seg_automask_image_min_area = gr.Number(
                            value=0,
                            label="Min Area",
                        )
                    with gr.Row():
                        with gr.Column():
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

                seg_automask_image_predict = gr.Button(value="Generator")

            with gr.Column():
                output_image = gr.Image()

        seg_automask_image_predict.click(
            fn=automask_image_app,
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
                        seg_automask_video_min_area = gr.Number(
                            value=1000,
                            label="Min Area",
                        )

                    with gr.Row():
                        with gr.Column():
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

                seg_automask_video_predict = gr.Button(value="Generator")
            with gr.Column():
                output_video = gr.Video()

        seg_automask_video_predict.click(
            fn=automask_video_app,
            inputs=[
                seg_automask_video_file,
                seg_automask_video_model_type,
                seg_automask_video_points_per_side,
                seg_automask_video_points_per_batch,
                seg_automask_video_min_area,
            ],
            outputs=[output_video],
        )


def sahi_app():
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                sahi_image_file = gr.Image(type="filepath").style(height=260)
                sahi_autoseg_model_type = gr.Dropdown(
                    choices=[
                        "vit_h",
                        "vit_l",
                        "vit_b",
                    ],
                    value="vit_l",
                    label="Sam Model Type",
                )

                with gr.Row():
                    with gr.Column():
                        sahi_model_type = gr.Dropdown(
                            choices=[
                                "yolov5",
                                "yolov8",
                            ],
                            value="yolov5",
                            label="Detector Model Type",
                        )
                        sahi_image_size = gr.Slider(
                            minimum=0,
                            maximum=1600,
                            step=32,
                            value=640,
                            label="Image Size",
                        )

                        sahi_overlap_width = gr.Slider(
                            minimum=0,
                            maximum=1,
                            step=0.1,
                            value=0.2,
                            label="Overlap Width",
                        )

                        sahi_slice_width = gr.Slider(
                            minimum=0,
                            maximum=640,
                            step=32,
                            value=256,
                            label="Slice Width",
                        )

                    with gr.Row():
                        with gr.Column():
                            sahi_model_path = gr.Dropdown(
                                choices=["yolov5l.pt", "yolov5l6.pt", "yolov8l.pt", "yolov8x.pt"],
                                value="yolov5l6.pt",
                                label="Detector Model Path",
                            )

                            sahi_conf_th = gr.Slider(
                                minimum=0,
                                maximum=1,
                                step=0.1,
                                value=0.2,
                                label="Confidence Threshold",
                            )
                            sahi_overlap_height = gr.Slider(
                                minimum=0,
                                maximum=1,
                                step=0.1,
                                value=0.2,
                                label="Overlap Height",
                            )
                            sahi_slice_height = gr.Slider(
                                minimum=0,
                                maximum=640,
                                step=32,
                                value=256,
                                label="Slice Height",
                            )
                sahi_image_predict = gr.Button(value="Generator")

            with gr.Column():
                output_image = gr.Image()

        sahi_image_predict.click(
            fn=sahi_autoseg_app,
            inputs=[
                sahi_image_file,
                sahi_autoseg_model_type,
                sahi_model_type,
                sahi_model_path,
                sahi_conf_th,
                sahi_image_size,
                sahi_slice_height,
                sahi_slice_width,
                sahi_overlap_height,
                sahi_overlap_width,
            ],
            outputs=[output_image],
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
                with gr.Tab("SAHI"):
                    sahi_app()

    app.queue(concurrency_count=1)
    app.launch(debug=True, enable_queue=True)


if __name__ == "__main__":
    metaseg_app()
