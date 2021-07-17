import datetime
import cv2
from enum import Enum

def read_costco(response):
    lines = get_sorted_lines(response)
    receipt_rows = []
    for line in lines:
        texts = [i[2] for i in line]
        texts = ''.join(texts)
        receipt_rows.append(texts)
    total_index = receipt_rows.index('合計') + 1
    total_price = receipt_rows[total_index]
    price = total_price.replace(',', '').replace('.', '')
    costco_date = receipt_rows[-3].split('/')[:2]
    costco_date_year = receipt_rows[-3].split('/')[2][:2]
    costco_date.append(costco_date_year)
    date_dt = datetime.datetime.strptime('/'.join(costco_date), '%m/%d/%y')
    date_dt = date_dt.strftime('%Y/%m/%d')
    return price, date_dt




def draw_boxes(input_file, bounds):
    img = cv2.imread(input_file, cv2.IMREAD_COLOR)
    for bound in bounds:
      p1 = (bound.vertices[0].x, bound.vertices[0].y) # top left
      p2 = (bound.vertices[1].x, bound.vertices[1].y) # top right
      p3 = (bound.vertices[2].x, bound.vertices[2].y) # bottom right
      p4 = (bound.vertices[3].x, bound.vertices[3].y) # bottom left
      cv2.line(img, p1, p2, (0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
      cv2.line(img, p2, p3, (0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
      cv2.line(img, p3, p4, (0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
      cv2.line(img, p4, p1, (0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
    return img


class FeatureType(Enum):
    PAGE = 1
    BLOCK = 2
    PARA = 3
    WORD = 4
    SYMBOL = 5



def get_document_bounds(response, feature):
    document = response.full_text_annotation
    bounds = []
    for page in document.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    for symbol in word.symbols:
                        if (feature == FeatureType.SYMBOL):
                          bounds.append(symbol.bounding_box)
                    if (feature == FeatureType.WORD):
                        bounds.append(word.bounding_box)
                if (feature == FeatureType.PARA):
                    bounds.append(paragraph.bounding_box)
            if (feature == FeatureType.BLOCK):
                bounds.append(block.bounding_box)
    return bounds


def get_sorted_lines(response):
    document = response.full_text_annotation
    bounds = []
    for page in document.pages:
      for block in page.blocks:
        for paragraph in block.paragraphs:
          for word in paragraph.words:
            for symbol in word.symbols:
              x = symbol.bounding_box.vertices[0].x
              y = symbol.bounding_box.vertices[0].y
              text = symbol.text
              bounds.append([x, y, text, symbol.bounding_box])
    bounds.sort(key=lambda x: x[1])
    old_y = -1
    line = []
    lines = []
    threshold = 1
    for bound in bounds:
      x = bound[0]
      y = bound[1]
      if old_y == -1:
        old_y = y
      elif old_y-threshold <= y <= old_y+threshold:
        old_y = y
      else:
        old_y = -1
        line.sort(key=lambda x: x[0])
        lines.append(line)
        line = []
      line.append(bound)
    line.sort(key=lambda x: x[0])
    lines.append(line)
    return lines