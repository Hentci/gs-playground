from PIL import Image

def resize_image(input_path, output_path=None):
    """
    將RGBA圖片調整為1237x822大小
    
    參數:
    input_path: 輸入圖片的路徑
    output_path: 輸出圖片的路徑，如果未指定則在原檔案名稱後加上"_resized"
    """
    # 如果沒有指定輸出路徑，則在原檔案名稱後加上"_resized"
    if output_path is None:
        file_name = os.path.splitext(input_path)[0]
        file_ext = ".png"  # 強制使用PNG格式以支援透明通道
        output_path = f"{file_name}_resized{file_ext}"
    
    # 開啟圖片
    with Image.open(input_path) as img:
        # 調整圖片大小，使用LANCZOS重採樣方法以獲得較好的品質
        resized_img = img.resize((1297, 840), Image.LANCZOS)
        
        # 檢查圖片模式
        if resized_img.mode == 'RGBA':
            if output_path.lower().endswith('.jpg') or output_path.lower().endswith('.jpeg'):
                # 如果要儲存為JPG，需要先轉換為RGB
                # 建立白色背景
                white_bg = Image.new('RGB', resized_img.size, (255, 255, 255))
                # 將RGBA圖片貼到白色背景上
                white_bg.paste(resized_img, mask=resized_img.split()[3])  # 使用alpha通道作為遮罩
                # 儲存結果
                white_bg.save(output_path, quality=95)
            else:
                # 如果是PNG格式，直接儲存
                resized_img.save(output_path)
        else:
            # 如果不是RGBA格式，直接儲存
            resized_img.save(output_path, quality=95)
            
        print(f"圖片大小調整完成！檔案已儲存至: {output_path}")
        print(f"新圖片尺寸: {resized_img.size}")

# 使用範例
import os
input_path = "/home/hentci/code/gs-playground/tools/2024-11-04_23.34.04-removebg.png"
resize_image(input_path, "/project/hentci/mip-nerf-360/trigger_garden_fox/fox.png")