import sys
import os
from enum import Enum, auto
from collections import defaultdict, OrderedDict
from tkinter import Tk, filedialog, messagebox, simpledialog, Text, Scrollbar, Frame, Button, Toplevel
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch