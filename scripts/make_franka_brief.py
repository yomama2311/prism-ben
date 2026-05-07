"""make_franka_brief.py — generate a PDF brief on the real-robot ROS communication
state of the PRISM-JEPA project.

Output: /home/yuhai/project/prism-jepa/franka_ros_brief.pdf

Run:
    .venv/bin/python scripts/make_franka_brief.py
"""
from __future__ import annotations

import io
import os
from datetime import date

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


# ============================================================
# 1. Matplotlib diagrams — saved to in-memory PNG bytes
# ============================================================

PRIMARY = "#1f4e79"     # deep blue — workstation / our side
ACCENT  = "#c6504e"     # rust — robot / dangerous
NEUTRAL = "#5b6770"     # slate — control PC / intermediary
GREEN   = "#2e7d32"     # working / verified
LIGHT_BG = "#f0f3f7"


def _save_fig_to_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def fig_architecture() -> bytes:
    """Three-machine architecture: workstation -- WiFi -- control PC -- FCI -- robot."""
    fig, ax = plt.subplots(figsize=(11, 5.5))

    boxes = [
        # (x, y, w, h, label, color)
        (0.5, 1.5, 3.4, 3.0,
         "Your Workstation\n"
         "(RTX 5090, Ubuntu 22.04)\n"
         "192.168.1.30/24\n\n"
         "• conda env: franka\n"
         "  (RoboStack ROS1 Noetic)\n"
         "• catkin ws: ~/franka_ws\n"
         "  franka_msgs compiled\n"
         "• rospy client\n"
         "  (subscribe + publish)",
         PRIMARY),
        (4.7, 1.5, 3.4, 3.0,
         "Lab Control PC\n"
         "(RT laptop, PREEMPT_RT)\n"
         "192.168.1.10\n\n"
         "• roscore (master)\n"
         "• franka_control\n"
         "  → FCI @ 1 kHz\n"
         "• franka_state_controller\n"
         "  → publish 30 Hz\n"
         "• cartesian_impedance_\n"
         "  example_controller\n"
         "  → subscribe equilibrium_pose",
         NEUTRAL),
        (8.9, 1.5, 2.4, 3.0,
         "Franka Panda\n\n\n"
         "• 7-DoF arm\n"
         "• FCI control I/F\n"
         "• 1 kHz internal\n"
         "  control loop\n"
         "• Reflex stop on\n"
         "  collision /\n"
         "  velocity limit",
         ACCENT),
    ]

    for (x, y, w, h, label, color) in boxes:
        box = FancyBboxPatch((x, y), w, h,
                             boxstyle="round,pad=0.06,rounding_size=0.15",
                             linewidth=1.5, edgecolor=color,
                             facecolor=LIGHT_BG)
        ax.add_patch(box)
        ax.text(x + w/2, y + h - 0.2, label.split('\n')[0],
                ha='center', va='top', fontsize=11, fontweight='bold',
                color=color)
        rest = '\n'.join(label.split('\n')[1:])
        ax.text(x + w/2, y + h - 0.6, rest,
                ha='center', va='top', fontsize=8, color='#222')

    # Arrows + labels: workstation -- WiFi --> control PC
    arrow1 = FancyArrowPatch((3.9, 3.3), (4.7, 3.3),
                             arrowstyle='<->', mutation_scale=18,
                             linewidth=2, color=GREEN)
    ax.add_patch(arrow1)
    ax.text(4.3, 3.65, "WiFi (frankaros_5g)\nROS over TCP\n~30 Hz state stream",
            ha='center', va='bottom', fontsize=8.5, color=GREEN, fontweight='bold')

    # Arrow: control PC -- FCI --> robot
    arrow2 = FancyArrowPatch((8.1, 3.3), (8.9, 3.3),
                             arrowstyle='<->', mutation_scale=18,
                             linewidth=2, color=ACCENT)
    ax.add_patch(arrow2)
    ax.text(8.5, 3.65, "FCI\n1 kHz",
            ha='center', va='bottom', fontsize=8.5, color=ACCENT, fontweight='bold')

    # Below: data-flow caption
    ax.text(6, 0.8, "Read path  (verified ✓):  Robot → FCI → state_controller → ROS msg → wlp7s0 → "
                    "rospy subscriber → hello_robot.py",
            ha='center', va='center', fontsize=9, color=GREEN, fontweight='bold')
    ax.text(6, 0.45, "Write path (NOT yet):  PRISM-MPPI → PoseStamped → /equilibrium_pose → "
                    "cartesian_impedance_controller → FCI → robot",
            ha='center', va='center', fontsize=9, color="#888")

    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title("Real-robot architecture: 3 machines, 2 network hops",
                 fontsize=13, fontweight='bold', pad=10)

    return _save_fig_to_bytes(fig)


def fig_ros_layers() -> bytes:
    """Vertical OSI-style layer stack showing what's verified vs pending."""
    fig, ax = plt.subplots(figsize=(8, 6))

    layers = [
        # (label, status, detail)
        ("L7  ROS Application",     "verified",
         "hello_robot.py subscribes /franka_state_controller/franka_states\n"
         "received 91 msgs in 3 s @ 30 Hz, 0 dropped\n"
         "decoded q, O_T_EE, robot_mode, errors"),
        ("L7  ROS Master / Topics", "verified",
         "rostopic list returns ~50 lab topics including:\n"
         "  /cartesian_impedance_example_controller/equilibrium_pose\n"
         "  /franka_control/error_recovery/*  /franka_gripper/*"),
        ("L7  Message marshalling", "verified",
         "franka_msgs catkin-built (Python 3.11 site-packages)\n"
         "FrankaState, ErrorRecoveryAction, SetCartesianImpedance, etc."),
        ("L4  TCP to ROS master",    "verified",
         "tcp connection to 192.168.1.10:11311 establishes (after\n"
         "Xiao Hu starts franka_run.py); before that: Connection refused"),
        ("L3  IP routing",           "verified",
         "ping 192.168.1.10  → 0% loss, RTT 28-209 ms (high variance)\n"
         "static route via wlp7s0 → 192.168.1.30 (direct subnet)"),
        ("L2  WiFi association",    "verified",
         "wlp7s0 connected to frankaros_5g\n"
         "static IPv4 192.168.1.30/24 (DHCP would land .0.x — wrong subnet!)"),
        ("PUBLISH (write) path",     "pending",
         "Need to write: PoseStamped publisher → /equilibrium_pose\n"
         "+ ErrorRecoveryAction client (mandatory)\n"
         "+ rate limit, EE workspace bounds, smooth init from current pose"),
    ]

    n = len(layers)
    h = 0.85
    gap = 0.05
    for i, (label, status, detail) in enumerate(layers):
        y = (n - 1 - i) * (h + gap)
        color = GREEN if status == "verified" else "#b75e2c"
        bg = "#e6f4ea" if status == "verified" else "#fcefe5"
        marker = "✓" if status == "verified" else "○"

        box = FancyBboxPatch((0, y), 8, h,
                             boxstyle="round,pad=0.04,rounding_size=0.08",
                             linewidth=1.2, edgecolor=color,
                             facecolor=bg)
        ax.add_patch(box)

        ax.text(0.15, y + h - 0.18, marker,
                ha='left', va='top', fontsize=14, fontweight='bold', color=color)
        ax.text(0.55, y + h - 0.18, label,
                ha='left', va='top', fontsize=10.5, fontweight='bold', color=color)
        ax.text(0.55, y + 0.42, detail,
                ha='left', va='top', fontsize=7.5, color="#222")

    ax.set_xlim(-0.1, 8.1)
    ax.set_ylim(-0.1, n * (h + gap))
    ax.axis('off')
    ax.set_title("ROS communication stack — verification status (today)",
                 fontsize=12, fontweight='bold', pad=10)
    return _save_fig_to_bytes(fig)


def fig_data_flow() -> bytes:
    """Operational data flow during PRISM-MPPI deployment."""
    fig, ax = plt.subplots(figsize=(11, 5))

    # Left column: workstation
    # Right column: control PC + robot
    pieces = [
        # (x, y, w, h, label, color)
        (0.5, 4.0, 2.5, 0.7, "RGB camera\n(ROS topic)", "#666"),
        (0.5, 2.9, 2.5, 0.7, "JEPA encoder\n(z_t, z_g)", PRIMARY),
        (0.5, 1.8, 2.5, 0.7, "Prior head\n(μ, σ)", PRIMARY),
        (0.5, 0.7, 2.5, 0.7, "PRISM-MPPI\nplanner @ 10 Hz", PRIMARY),

        (5.5, 4.0, 2.5, 0.7, "franka_state_\ncontroller @ 30 Hz", NEUTRAL),
        (5.5, 2.9, 2.5, 0.7, "/franka_state_\ncontroller/franka_states", "#7d8b96"),
        (5.5, 1.8, 2.5, 0.7, "cartesian_impedance\n_example_controller", NEUTRAL),
        (5.5, 0.7, 2.5, 0.7, "FCI 1 kHz → Franka\n(impedance tracking)", ACCENT),
    ]

    for (x, y, w, h, label, color) in pieces:
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.05",
                                    linewidth=1.2, edgecolor=color,
                                    facecolor=LIGHT_BG))
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                fontsize=8.5, color=color, fontweight='bold')

    # Read-back arrows (right -> left) at top
    for (sx, ex, y) in [(5.5, 3.0, 4.35)]:
        a = FancyArrowPatch((sx, y), (ex, y), arrowstyle='->',
                            mutation_scale=15, color=GREEN, linewidth=1.5)
        ax.add_patch(a)
    ax.text(4.25, 4.55, "obs (RGB)", ha='center', fontsize=8, color=GREEN)

    # State flow downward inside each side, then read-back
    ax.text(1.75, 3.65, "▼", ha='center', fontsize=10, color=PRIMARY)
    ax.text(1.75, 2.55, "▼", ha='center', fontsize=10, color=PRIMARY)
    ax.text(1.75, 1.45, "▼", ha='center', fontsize=10, color=PRIMARY)

    ax.text(6.75, 3.65, "▼", ha='center', fontsize=10, color=NEUTRAL)
    ax.text(6.75, 2.55, "▼", ha='center', fontsize=10, color=NEUTRAL)
    ax.text(6.75, 1.45, "▼", ha='center', fontsize=10, color=NEUTRAL)

    # Action publish arrow (left -> right) at bottom
    a = FancyArrowPatch((3.0, 1.0), (5.5, 1.0), arrowstyle='->',
                        mutation_scale=18, color=ACCENT, linewidth=2)
    ax.add_patch(a)
    ax.text(4.25, 1.45, "PoseStamped @ ~10 Hz\n(equilibrium_pose)",
            ha='center', fontsize=8, color=ACCENT, fontweight='bold')

    # State subscribe arrow (right -> left)
    a = FancyArrowPatch((5.5, 3.25), (3.0, 3.25), arrowstyle='->',
                        mutation_scale=18, color=GREEN, linewidth=2)
    ax.add_patch(a)
    ax.text(4.25, 3.55, "FrankaState @ 30 Hz\n(joint q, EE pose, errors)",
            ha='center', fontsize=8, color=GREEN, fontweight='bold')

    # Header labels
    ax.text(1.75, 5.0, "Your workstation",
            ha='center', fontsize=11, fontweight='bold', color=PRIMARY)
    ax.text(6.75, 5.0, "Lab control PC + Franka",
            ha='center', fontsize=11, fontweight='bold', color=NEUTRAL)

    ax.set_xlim(0, 9)
    ax.set_ylim(0, 5.5)
    ax.axis('off')
    ax.set_title("PRISM real-time data flow during PushT deployment",
                 fontsize=12, fontweight='bold', pad=8)
    return _save_fig_to_bytes(fig)


def fig_safety_path() -> bytes:
    """Decision tree: how to proceed with publishers safely."""
    fig, ax = plt.subplots(figsize=(9, 5.2))

    nodes = [
        # (x, y, w, h, label, color)
        (3.5, 4.4, 2.5, 0.7,
         "Wrote a publisher\nthat sends action?", "#444"),
        (1.0, 3.0, 2.5, 0.7,
         "Receive Xiao Hu's\nfranka_run.py first", GREEN),
        (6.0, 3.0, 2.5, 0.7,
         "Read his publish\nsemantics + frame", GREEN),
        (3.5, 1.6, 2.5, 0.7,
         "Implement matching\npublisher + ErrorRecovery", PRIMARY),
        (3.5, 0.3, 2.5, 0.7,
         "Test with Xiao Hu\non site, hand on E-stop", ACCENT),
    ]
    for (x, y, w, h, label, color) in nodes:
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.05",
                                    linewidth=1.5, edgecolor=color,
                                    facecolor=LIGHT_BG))
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                fontsize=9, color=color, fontweight='bold')

    # arrows
    arr = lambda x1,y1,x2,y2,col: ax.add_patch(
        FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='->',mutation_scale=14,color=col,linewidth=1.3))
    arr(3.5, 4.4, 2.25, 3.7, "#444")  # to first
    arr(6.0, 4.4, 7.25, 3.7, "#444")  # to second
    arr(2.25, 3.0, 3.5, 2.3, GREEN)
    arr(7.25, 3.0, 6.0, 2.3, GREEN)
    arr(4.75, 1.6, 4.75, 1.0, PRIMARY)

    # warning bar
    ax.text(4.75, 5.0, "⚠ Safety gate: ANY publisher must pass these before being run on the real robot",
            ha='center', fontsize=10, color=ACCENT, fontweight='bold')

    ax.set_xlim(0, 9.5)
    ax.set_ylim(0, 5.5)
    ax.axis('off')
    return _save_fig_to_bytes(fig)


# ============================================================
# 2. ReportLab PDF assembly
# ============================================================

def make_pdf(out_path: str) -> None:
    doc = SimpleDocTemplate(out_path, pagesize=letter,
                            leftMargin=0.6*inch, rightMargin=0.6*inch,
                            topMargin=0.6*inch, bottomMargin=0.6*inch)
    styles = getSampleStyleSheet()

    h1 = ParagraphStyle("h1", parent=styles["Heading1"],
                        fontSize=18, spaceAfter=8, textColor=colors.HexColor(PRIMARY))
    h2 = ParagraphStyle("h2", parent=styles["Heading2"],
                        fontSize=13, spaceBefore=10, spaceAfter=4,
                        textColor=colors.HexColor(PRIMARY))
    h3 = ParagraphStyle("h3", parent=styles["Heading3"],
                        fontSize=11, spaceBefore=6, spaceAfter=2,
                        textColor=colors.HexColor(NEUTRAL))
    body = ParagraphStyle("body", parent=styles["BodyText"],
                          fontSize=9.5, leading=12, alignment=TA_LEFT,
                          spaceAfter=4)
    code = ParagraphStyle("code", parent=styles["Code"],
                          fontSize=8.5, leading=10,
                          textColor=colors.HexColor("#444"),
                          backColor=colors.HexColor("#f6f6f6"))
    callout = ParagraphStyle("callout", parent=body, fontSize=9.5, leading=12,
                             textColor=colors.HexColor(ACCENT))
    sub = ParagraphStyle("sub", parent=body, fontSize=8.5, textColor=colors.HexColor("#777"))

    story = []

    # Title block
    story.append(Paragraph("PRISM-JEPA Real-Robot ROS Communication", h1))
    story.append(Paragraph(
        f"Status briefing • {date.today().isoformat()} • Author: Yuhai • "
        f"Sister project: <i>prism-jepa</i> (LeWM substrate)",
        sub))
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "<b>One-line status.</b> Today the full read-only ROS communication path from this workstation to "
        "the lab Franka was verified: WiFi → IP routing → ROS master → topic subscribe → 30 Hz "
        "real robot state. The publish (write) path is intentionally not yet implemented — it requires "
        "Xiao Hu's <i>franka_run.py</i> as a reference for lab convention before we can safely send "
        "commands.",
        body))

    # ---- §1 Architecture ----
    story.append(Paragraph("1. Architecture: three machines, two network hops", h2))
    story.append(Image(io.BytesIO(fig_architecture()), width=7.0*inch, height=3.5*inch))
    story.append(Paragraph(
        "Our workstation is a <b>ROS client</b>. Real-time torque control is the lab control "
        "laptop's job (PREEMPT_RT-patched kernel + libfranka over FCI). We publish high-level EE "
        "pose targets at ~10 Hz over WiFi; the lab laptop tracks them at 1 kHz internally and "
        "publishes 30 Hz state back. We never need RT scheduling on this side — that's the whole "
        "point of separating concerns.",
        body))

    # ---- §2 ROS layer status ----
    story.append(Paragraph("2. ROS communication stack — verification status", h2))
    story.append(Image(io.BytesIO(fig_ros_layers()), width=6.5*inch, height=4.9*inch))
    story.append(Paragraph(
        "All read-side layers are verified end-to-end with real Franka data. The single remaining "
        "row is the publish path — by design, since writing a publisher means sending commands "
        "the robot will execute, and we need Xiao Hu's reference script first to align with lab "
        "convention (frame, rate, smoothing).",
        body))

    story.append(PageBreak())

    # ---- §3 Network setup ----
    story.append(Paragraph("3. Network setup — what we configured", h2))
    netconfig_data = [
        ["Step", "What", "Value", "Constraint"],
        ["1", "Connect to lab WiFi", "frankaros_5g", "must be physically in the lab"],
        ["2", "WiFi IPv4 (manual, NOT DHCP)", "192.168.1.30/24", "must equal $ROS_HOSTNAME"],
        ["3", "ROS env vars", "MASTER=192.168.1.10:11311\nHOSTNAME=192.168.1.30",
         "set by ~/franka_ws/setup_franka.sh"],
        ["4", "L3 reachability", "ping 192.168.1.10", "0% loss expected"],
        ["5", "L7 reachability", "rostopic list", "needs lab roscore running"],
    ]
    t = Table(netconfig_data, colWidths=[0.4*inch, 1.7*inch, 2.1*inch, 2.5*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(PRIMARY)),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 8.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("GRID",       (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f6f6f6")]),
        ("VALIGN",     (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(t)
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<b>The classic gotcha</b> we hit today: WiFi DHCP gave us 192.168.0.102 (different "
        "subnet from the control PC at 192.168.1.10), so ping failed even though the WiFi was "
        "connected. The fix is the manual static-IP step (item 2 above) — explicitly required "
        "by the lab email but easy to miss.",
        body))

    # ---- §4 What's running on the lab control PC ----
    story.append(Paragraph("4. What's running on the lab control PC (after Xiao Hu starts the stack)", h2))
    rosstack_data = [
        ["Component", "Status today", "What it does", "We use it as"],
        ["roscore", "✓ on 192.168.1.10:11311", "ROS message bus / master",
         "the address we set $ROS_MASTER_URI to"],
        ["franka_control", "✓ running", "talks to robot via FCI at 1 kHz",
         "we don't touch it directly"],
        ["franka_state_controller", "✓ publishing 30 Hz",
         "publishes joint q, EE pose, errors",
         "subscribe /franka_state_controller/franka_states"],
        ["cartesian_impedance_\nexample_controller", "✓ loaded",
         "tracks an EE pose target via impedance",
         "publish PoseStamped to .../equilibrium_pose"],
        ["error_recovery action server", "✓ on /franka_control/error_recovery/*",
         "clears reflex / collision latch",
         "MUST call after any reflex; not optional"],
        ["franka_gripper", "✓ on /franka_gripper/*",
         "open/close/grasp action interface",
         "PushT doesn't need this"],
    ]
    t = Table(rosstack_data, colWidths=[1.5*inch, 1.4*inch, 1.9*inch, 2.2*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(PRIMARY)),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("GRID",       (0, 0), (-1, -1), 0.4, colors.lightgrey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f6f6f6")]),
        ("VALIGN",     (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(t)

    story.append(PageBreak())

    # ---- §5 Today's data ----
    story.append(Paragraph("5. Real-robot state we read today", h2))
    state_data = [
        ["Field", "Value", "Interpretation"],
        ["q (7 joints, rad)",
         "[-0.002, -0.785, 0.000, -2.364,\n -0.003, +1.572, +0.777]",
         "Standard Franka home pose\n[0, -π/4, 0, -3π/4, 0, π/2, π/4]"],
        ["EE position (m)", "(+0.306, -0.001, +0.483)",
         "30 cm in front of base, 48 cm above table"],
        ["robot_mode", "2 (MOVE)", "Motor power on, ready to receive commands"],
        ["control_command_success_rate", "1.000", "Controller is happy"],
        ["current_errors", "False", "No latched errors"],
        ["State publish rate measured", "~30 Hz over 3 s, 91 msgs",
         "Matches franka_state_controller default; 0 dropped"],
        ["WiFi RTT", "min 28 ms / max 209 ms / mdev 81 ms",
         "High variance — flag for closed-loop testing later"],
    ]
    t = Table(state_data, colWidths=[1.6*inch, 2.4*inch, 3.0*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(PRIMARY)),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 8.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("GRID",       (0, 0), (-1, -1), 0.4, colors.lightgrey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f6f6f6")]),
        ("VALIGN",     (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(t)

    # ---- §6 Operational data flow during deployment ----
    story.append(Paragraph("6. Operational data flow during PushT deployment", h2))
    story.append(Image(io.BytesIO(fig_data_flow()), width=7.0*inch, height=3.2*inch))
    story.append(Paragraph(
        "<b>Read leg (verified):</b> camera RGB → JEPA encode → (z_t, z_g) → prior head → (μ, σ) "
        "→ PRISM-MPPI planner. State subscription confirmed at 30 Hz today. "
        "<b>Write leg (pending):</b> planner output → PoseStamped @ ~10 Hz → "
        "/cartesian_impedance_example_controller/equilibrium_pose → tracked at 1 kHz on the "
        "lab side → robot moves.",
        body))

    story.append(PageBreak())

    # ---- §7 What's pending and why ----
    story.append(Paragraph("7. What's intentionally not yet done", h2))
    pending_data = [
        ["Pending item", "Why we didn't do it today"],
        ["Publisher to /equilibrium_pose",
         "Sending a command moves the robot. We need Xiao Hu's franka_run.py "
         "first to align frames, rate, and smoothing convention."],
        ["ErrorRecoveryAction client",
         "Mandatory companion to any publisher: Franka latches errors on collision "
         "and refuses new commands until ErrorRecovery is sent. Skipping this "
         "would brick the robot for the next user."],
        ["Read Xiao Hu's franka_run.py",
         "He hasn't shared the file yet (his keyboard-control reference). "
         "We don't guess at lab convention."],
        ["Camera ROS topic",
         "Need to ask Xiao Hu whether the lab has an RGB camera already publishing, "
         "or if we set one up. PRISM needs RGB obs to feed JEPA."],
        ["Real-robot training data",
         "Sim-trained JEPA + world model + prior head will not transfer. "
         "Plan: 200 real teleop demos → re-train stack → eval. ~2 weeks."],
    ]
    t = Table(pending_data, colWidths=[2.0*inch, 5.0*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(ACCENT)),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("GRID",       (0, 0), (-1, -1), 0.4, colors.lightgrey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#fcefe5")]),
        ("VALIGN",     (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(t)

    # ---- §8 Safety gating ----
    story.append(Paragraph("8. Safety gating before any publisher runs", h2))
    story.append(Image(io.BytesIO(fig_safety_path()), width=6.0*inch, height=3.2*inch))
    story.append(Paragraph(
        "This is the gate every PRISM publisher passes through before being run against the real "
        "Franka. The ErrorRecovery client must be implemented first; the EE workspace bounds must "
        "be coded in; the first PoseStamped sent must be from the robot's current EE pose to avoid "
        "an instantaneous tracking jump; and Xiao Hu (or another lab member with E-stop access) "
        "should be present for the first 5–10 closed-loop trials.",
        body))

    # ---- §9 Likely Q&A ----
    story.append(Paragraph("9. Anticipated questions and prepared answers", h2))
    qa_data = [
        ["Q", "A"],
        ["What's verified end-to-end today?",
         "L2 (WiFi) → L3 (IP) → L4 (TCP to ROS master) → L7 (subscribe FrankaState). "
         "Real robot at home pose, MOVE mode, 30 Hz state stream, 0 dropped messages."],
        ["Why no publisher yet?",
         "Intentional. Need Xiao Hu's franka_run.py for lab convention; need to "
         "co-implement ErrorRecoveryAction client (mandatory); need EE workspace "
         "bounds coded in. Estimated ~3-5 days to a safe first closed-loop hello-world."],
        ["Is WiFi reliable enough?",
         "30 Hz state with 0 packet loss today, but RTT variance is high (28-209 ms). "
         "Flagged as risk for closed-loop publishing; mitigations are wired Ethernet, "
         "5 GHz only, or reducing publish rate."],
        ["Why not run RT control on our workstation?",
         "GPU compute (JEPA + MPPI) preempts CPU and would trigger Franka safety stops. "
         "Plus we'd need a PREEMPT_RT kernel. The lab control laptop already does this; "
         "splitting concerns is the standard topology."],
        ["What's the timeline to first PushT closed-loop?",
         "Receive Xiao Hu's script (1-2 days) + align publish semantics (½ day) + "
         "minimal publisher with error_recovery (1 day) + on-site test (½ day) = ~3-5 days."],
        ["Sim-to-real gap?",
         "The sim-trained JEPA encoder will not generalize to real Franka pixels. Need "
         "~200 real teleop demos and re-train. Adds ~2 weeks to the experiment timeline; "
         "is the dominant cost of going to real."],
    ]
    t = Table(qa_data, colWidths=[1.7*inch, 5.3*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(PRIMARY)),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("GRID",       (0, 0), (-1, -1), 0.4, colors.lightgrey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f6f6f6")]),
        ("VALIGN",     (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(t)

    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "<i>Files referenced in this brief:</i> "
        "~/franka_ws/setup_franka.sh, ~/franka_ws/hello_robot.py, "
        "~/franka_ws/mock_franka_publisher.py, ~/franka_ws/src/franka_ros/.",
        sub))

    doc.build(story)


if __name__ == "__main__":
    out = "/home/yuhai/project/prism-jepa/franka_ros_brief.pdf"
    make_pdf(out)
    size_kb = os.path.getsize(out) / 1024
    print(f"[brief] wrote {out} ({size_kb:.1f} KB)")
