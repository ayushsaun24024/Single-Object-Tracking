import os
import gradio as gr
from inference import ObjectTrackerInference


tracker = ObjectTrackerInference(model_dir='models')


def track_object(video, x, y, width, height):
    try:
        if video is None:
            return None
        
        initial_bbox = [int(x), int(y), int(width), int(height)]
        
        output_path = 'tracked_output.mp4'
        result = tracker.track_video(video, initial_bbox, output_path, fps=30)
        
        return result
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

with gr.Blocks(title="UAV Object Tracker") as demo:
    
    gr.Markdown("# 🎯 UAV Single Object Tracker")
    gr.Markdown("Upload a video and specify the initial bounding box to track an object.")
    
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Upload Video")
            
            gr.Markdown("### Initial Bounding Box Coordinates")
            with gr.Row():
                x_input = gr.Number(label="X (top-left)", value=100)
                y_input = gr.Number(label="Y (top-left)", value=100)
            with gr.Row():
                w_input = gr.Number(label="Width", value=50)
                h_input = gr.Number(label="Height", value=50)
            
            track_btn = gr.Button("Track Object", variant="primary")
        
        with gr.Column():
            video_output = gr.Video(label="Tracked Output")
    
    gr.Markdown("### 📖 Instructions")
    gr.Markdown("""
    1. Upload your video file
    2. Enter the initial bounding box coordinates (x, y, width, height) for the first frame
    3. Click 'Track Object' to process
    4. Download the tracked video from the output
    """)

    track_btn.click(
        fn=track_object,
        inputs=[video_input, x_input, y_input, w_input, h_input],
        outputs=video_output
    )

if __name__ == "__main__":
    demo.launch()
