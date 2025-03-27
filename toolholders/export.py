
from typing import Sequence
import math
import ezdxf
from ezdxf import units
from ezdxf.enums import TextEntityAlignment
from .shapes import Shape, Group, ShapeType, Text
import ezdxf.colors
import os.path

anchor_to_alignment = {
    "mm": TextEntityAlignment.MIDDLE_CENTER,
    "lt": TextEntityAlignment.TOP_LEFT,
}

def export_dxf(shapes: Sequence[Shape | Group], filename: str):
    doc = ezdxf.new(dxfversion="R2010", units=units.MM)
    msp = doc.modelspace()
    fonts = {}

    def add_shape(shape: Shape | Group | Text):
        if isinstance(shape, Text):
            shape = shape.to_curves(max_error=0.01)

        if isinstance(shape, Shape):
            contour = shape.contour
            entity = msp.add_lwpolyline(contour, close=True)
            if shape.type == ShapeType.Cut:
                entity.dxf.layer = 'CUT'
                entity.dxf.color = ezdxf.colors.BLACK
            elif shape.type == ShapeType.EngraveOutline:
                entity.dxf.layer = 'ENGRAVE_OUTLINE'
                entity.dxf.color = ezdxf.colors.BLUE
            else:
                entity.dxf.layer = 'ENGRAVE_FILL'
                entity.dxf.color = ezdxf.colors.GREEN
        elif isinstance(shape, Text):
            if shape.font not in fonts:
                name = "font-" + str(len(fonts))
                doc.styles.new(name, {"font" : os.path.basename(str(shape.font.path))})
                fonts[shape.font] = name
            
            entity = msp.add_text(
                shape.text,
                dxfattribs={'style': fonts[shape.font]},
                height=shape.font.size,
                rotation=math.degrees(shape.transform.rotation())
            )
            entity.set_placement(shape.transform.apply((0,0)), align=anchor_to_alignment[shape.anchor])
        else:
            for child in shape.children:
                add_shape(child)

    for shape in shapes:
        add_shape(shape)

    doc.saveas(filename)
