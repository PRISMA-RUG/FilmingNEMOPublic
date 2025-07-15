import video
import mediapipe_model
import landmarks
import gradio as gr

def get_and_cut_video(video_in, start_time, start_x, start_y, end_x, end_y):
    frame = video.get_single_frame(video_in, start_time, int(end_x), int(end_y),
                                   int(start_x), int(start_y))
    return frame

def main(video_in, start_time, end_time, start_x, start_y, end_x, end_y, model_type, target, axis):
    frames, fps, timestamps = video.load_video(file=video_in, start=start_time,
                                                         end=end_time, crop_x=int(start_x), crop_y=int(start_y),
                                                         width=int(end_x), height=int(end_y))
    landmarker = mediapipe_model.load_model_mp(type=model_type)
    keypoints = landmarks.get_landmarks(frames, landmarker, timestamps)
    frames = landmarks.paint_landmarks(frames, keypoints)
    figx = landmarks.plot_landmarks_over_time(keypoints, "x", mediapipe_model.MEDIAPIPE_ESTIMATOR_KEYPOINTS)
    figy = landmarks.plot_landmarks_over_time(keypoints, "y", mediapipe_model.MEDIAPIPE_ESTIMATOR_KEYPOINTS)
    figz = landmarks.plot_landmarks_over_time(keypoints, "z", mediapipe_model.MEDIAPIPE_ESTIMATOR_KEYPOINTS)
    video.save_video(frames, "videos/out.mp4", fps=fps)
    processed_dict = landmarks.landmarks_save(keypoints, "TEST", "TEST", model_type, start_time,
                                              end_time, end_x, end_y, start_x, start_y, fps)
    ccy, peaks = landmarks.peaks_and_consistency(processed_dict, target, mediapipe_model.MEDIAPIPE_ESTIMATOR_KEYPOINTS,
                                                 axis=axis)
    return ["videos/out.mp4", figx, figy, figz, ccy, peaks]

with gr.Blocks() as interface:
    with gr.Row():
        with gr.Column():
            video_in = gr.Video()
            target_landmark = gr.Dropdown(choices=list(mediapipe_model.MEDIAPIPE_ESTIMATOR_KEYPOINTS.values()),
                                          label="Target bodypart", type="index")
        with gr.Column():
            with gr.Row():
                start_x = gr.Textbox(value="0", label="Start Width (px)")
                end_x = gr.Textbox(value="1920", label="End Width (px)")
            with gr.Row():
                start_y = gr.Textbox(value="0", label="Start Height (px)")
                end_y = gr.Textbox(value="1080", label="End Height (px)")
            with gr.Row():
                start_time = gr.Textbox(value="00:00", label="Start Time (mm:ss)")
                end_time = gr.Textbox(value="00:30", label="End Time (mm:ss)")
            with gr.Row():
                model_type = gr.Radio(value="full", label="Model size", choices=["full", "lite"])
                target_axis = gr.Radio(value="x", label="Target Axis", choices=["x", "y", "z"])
    with gr.Row():
        with gr.Column():
            proceed1 = gr.Button("Preview Cropping")
            cut_frame = gr.Image(label="Crop Preview")
        with gr.Column():
            run_model = gr.Button("Run Landmarking Model")
    with gr.Row():
        video_out = gr.Video(label="Output Video", height=600)
    with gr.Row():
        with gr.Column():
            plotx = gr.Plot()
        with gr.Column():
            ploty = gr.Plot()
        with gr.Column():
            plotz = gr.Plot()
    with gr.Row():
        plot_peaks =  gr.Plot()
    with gr.Row():
        metrics = gr.Number(label="Consistency")

    proceed1.click(fn=get_and_cut_video,
                   inputs=[video_in, start_time, start_x, start_y, end_x, end_y],
                   outputs=cut_frame)

    run_model.click(fn=main,
                    inputs=[video_in, start_time, end_time, start_x, start_y, end_x, end_y,
                            model_type, target_landmark, target_axis],
                    outputs=[video_out, plotx, ploty, plotz, metrics, plot_peaks])



interface.launch()