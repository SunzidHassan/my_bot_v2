from PIL import Image, ImageDraw 
import random
import numpy as np


def run_example(task_prompt, image, text_input=None):
    """Generates segmentation results based on the input image and task prompt."""
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    generated_ids = model.generate(
      input_ids=inputs["input_ids"].cuda(),
      pixel_values=inputs["pixel_values"].cuda(),
      max_new_tokens=1024,
      early_stopping=False,
      do_sample=False,
      num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, 
        task=task_prompt, 
        image_size=(image.width, image.height)
    )

    return parsed_answer

colormap = ['indigo']

def draw_polygons(image, prediction, fill_mask=False):  
    """Draws segmentation masks with polygons on an image and returns the modified image."""
    draw = ImageDraw.Draw(image)

    for polygons, label in zip(prediction['polygons'], prediction['labels']):  
        color = random.choice(colormap)  
        fill_color = random.choice(colormap) if fill_mask else None  
          
        for _polygon in polygons:  
            _polygon = np.array(_polygon).reshape(-1, 2)  
            if len(_polygon) < 3:  
                print('Invalid polygon:', _polygon)  
                continue  
              
            _polygon = (_polygon * 1).reshape(-1).tolist()  # Adjust for scale if needed
              
            # Draw the polygon  
            if fill_mask:  
                draw.polygon(_polygon, outline=color, fill=fill_color)  
            else:  
                draw.polygon(_polygon, outline=color)  
              
            # Draw the label text  
            draw.text((_polygon[0] + 8, _polygon[1] + 2), label, fill=color)  
  
    return image  # Return the modified image

def calculate_polygon_area(polygon_points):
    """
    Calculate the area of a polygon using the Shoelace formula (also known as surveyor's formula).
    
    Args:
        polygon_points (list): List of [x, y] coordinates defining the polygon vertices
    
    Returns:
        float: Area of the polygon
    """
    n = len(polygon_points)
    area = 0.0
    
    for i in range(n):
        j = (i + 1) % n
        area += polygon_points[i][0] * polygon_points[j][1]
        area -= polygon_points[j][0] * polygon_points[i][1]
    
    area = abs(area) / 2.0
    return area

def get_segmentation_ratio(segmentation_results, image_width=640, image_height=480):
    """
    Calculate the ratio of segmented area to total image area.
    
    Args:
        segmentation_results (dict): Dictionary containing segmentation polygons
        image_width (int): Width of the image (default 640 based on environment)
        image_height (int): Height of the image (default 480 based on environment)
    
    Returns:
        float: Ratio of segmented area to total image area
    """
    # Extract polygon points
    polygons = segmentation_results['<REFERRING_EXPRESSION_SEGMENTATION>']['polygons']
    
    # Calculate total segmented area (sum of all polygon areas)
    total_segmented_area = sum(calculate_polygon_area(polygon) for polygon in polygons)
    
    # Calculate total image area
    total_image_area = image_width * image_height
    
    # Calculate and return ratio
    ratio = round(total_segmented_area / total_image_area, 4)
    
    return ratio


def segment_image(image_input: Image.Image, text_input="a cat") -> Image.Image:
    """Segments a given PIL Image based on the specified text input and returns the segmented image."""
    # Resize the image if necessary
    image = image_input.resize((64, 64))

    task_prompt = '<REFERRING_EXPRESSION_SEGMENTATION>'
    results = run_example(task_prompt, image, text_input=text_input)
    
    ratio = get_segmentation_ratio(results)
    output_image = draw_polygons(image.copy(), results['<REFERRING_EXPRESSION_SEGMENTATION>'], fill_mask=True)
    
    return output_image, ratio  # Return the segmented image