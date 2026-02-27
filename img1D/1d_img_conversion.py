import os
import argparse
from PIL import Image
from queue import Queue
from threading import Thread


def getBinaryData(filename):
    """
    Extract byte values from binary executable file and store them into a list.
    """
    with open(filename, 'rb') as f:
        return list(f.read())


def createGreyscaleImageHorizontal(filename, output_dir=None):
    """
    Create a horizontal greyscale image with height=1.
    """
    data = getBinaryData(filename)
    width = len(data)
    height = 1
    size = (width, height)
    save_file(filename, data, size, 'L', output_dir)


def createRGBImageHorizontal(filename, output_dir=None):
    """
    Create a horizontal RGB image with height=1.
    """
    data = getBinaryData(filename)

    # Pad data so it's divisible by 3
    if len(data) % 3 != 0:
        data += [0] * (3 - len(data) % 3)

    rgb_data = []
    for i in range(0, len(data), 3):
        rgb_data.append((data[i], data[i + 1], data[i + 2]))

    width = len(rgb_data)
    height = 1
    size = (width, height)
    save_file(filename, rgb_data, size, 'RGB', output_dir)


def save_file(input_file, data, size, image_type, output_dir):
    try:
        image = Image.new(image_type, size)
        image.putdata(data)

        # Determine output path
        name, _ = os.path.splitext(os.path.basename(input_file))
        out_subdir = image_type

        if output_dir:
            image_folder = os.path.join(output_dir, out_subdir)
        else:
            image_folder = os.path.join(os.path.dirname(input_file), out_subdir)

        os.makedirs(image_folder, exist_ok=True)
        imagename = os.path.join(image_folder, f"{name}_{image_type}_horizontal.png")

        image.save(imagename)
        print(f"The file {imagename} saved.")
    except Exception as err:
        print(f"Error saving file {input_file}: {err}")


def run(file_queue, output_dir, mode):
    while not file_queue.empty():
        filename = file_queue.get()
        if mode in ("L", "both"):
            createGreyscaleImageHorizontal(filename, output_dir)
        if mode in ("RGB", "both"):
            createRGBImageHorizontal(filename, output_dir)
        file_queue.task_done()


def main(input_dir, output_dir=None, mode="both", thread_number=4):
    file_queue = Queue()
    for root, _, files in os.walk(input_dir):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_queue.put(filepath)

    for _ in range(thread_number):
        thread = Thread(target=run, args=(file_queue, output_dir, mode))
        thread.daemon = True
        thread.start()

    file_queue.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert binary files to horizontal images")
    parser.add_argument('input_dir', help="Input directory containing binary files")
    parser.add_argument('--output_dir', help="Directory to save output images (optional)", default=None)
    parser.add_argument('--mode', choices=["L", "RGB", "both"], default="both",
                        help="Image type to generate: L (greyscale), RGB (color), or both")

    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.mode)
