import py5

STATE_WELCOME = 0
STATE_RESTING_INSTRUCTIONS = 1
STATE_RESTING_RECORDING = 2
STATE_NEUROFEEDBACK_INSTRUCTIONS = 3
STATE_END = 4

current_state = STATE_WELCOME
title_font = None
italic_font = None
body_font = None

NEUROFEEDBACK_INSTRUCTIONS_TEXT = (
    "In this session, you will see a tree on the screen.\n\n"
    "Your goal is to maintain a calm, relaxed focus."
    "When you succeed in relaxing, you might notice the tree begin to grow.\n\n"
    "Simply observe the tree, and let your internal state guide its changes."
)

def setup():
    global title_font, italic_font, body_font
    py5.full_screen(py5.P2D) 
    py5.smooth(8) 
    py5.no_cursor()


def draw():
    """ The core experiment loop, called every frame. """
    py5.background(0)
    
    if current_state == STATE_WELCOME:
        draw_resting_instructions() 
        
    elif current_state == STATE_RESTING_RECORDING:
        draw_resting_recording_stage()
        
    elif current_state == STATE_NEUROFEEDBACK_INSTRUCTIONS:
        draw_neurofeedback_instructions_screen()
        
    elif current_state == STATE_END:
        draw_end_screen()


def draw_text_prompt_at_bottom():
    """ Standard prompt to advance the stage. """
    italic_font = py5.create_font("Helvetica-Oblique", 20)
    py5.text_font(italic_font)
    py5.text("Press the space bar to continue.", py5.width / 2, py5.height * 0.85)

def draw_resting_instructions():
    title_font = py5.create_font("Helvetica-Bold", 32)
    body_font = py5.create_font("Helvetica", 24)
    italic_font = py5.create_font("Helvetica-Oblique", 20)

    py5.fill(255)
    py5.text_align(py5.CENTER, py5.CENTER)
    
    py5.text_font(title_font)
    py5.text("Welcome to ANTARES", py5.width / 2, py5.height * 0.18)
    
    
    py5.text_font(italic_font)
    py5.text("Advancing Neurofeedback in Tinnitus", py5.width / 2, py5.height * 0.25)
    
    py5.text_font(body_font)
    py5.text_leading(35) # Breathable line spacing
    instr = (
        "In the first stage, we will conduct a short resting-state recording.\n"
        "During this time, please keep your gaze fixed on the cross."
    )
    py5.text(instr, py5.width / 2, py5.height / 2)
    draw_text_prompt_at_bottom()

def draw_resting_recording_stage():
    py5.stroke(255)
    py5.stroke_weight(2)
    py5.line(py5.width/2 - 45, py5.height/2, py5.width/2 + 45, py5.height/2)
    py5.line(py5.width/2, py5.height/2 - 45, py5.width/2, py5.height/2 + 45)

def draw_neurofeedback_instructions_screen():
    title_font = py5.create_font("Helvetica-Bold", 32)
    body_font = py5.create_font("Helvetica", 24)
    italic_font = py5.create_font("Helvetica-Oblique", 20)

    py5.fill(255)
    py5.text_align(py5.CENTER, py5.CENTER)
    py5.text_font(title_font)
    py5.text("You finished the resting-state section", py5.width / 2, py5.height * 0.18)
    
    # 2. Transition (Italic font)
    py5.text_font(italic_font)
    py5.text("Now we will start the main neurofeedback session.", py5.width / 2, py5.height * 0.25)
    py5.text_font(body_font)
    py5.text_leading(40) 
    py5.text(NEUROFEEDBACK_INSTRUCTIONS_TEXT, py5.width / 2, py5.height * 0.55)
    draw_text_prompt_at_bottom()

def draw_end_screen():

    title_font = py5.create_font("Helvetica-Bold", 32)
    body_font = py5.create_font("Helvetica", 24)
    italic_font = py5.create_font("Helvetica-Oblique", 20)
    
    py5.fill(255)
    py5.text_align(py5.CENTER, py5.CENTER)
    py5.text_font(title_font)
    py5.text("Session Completed.", py5.width / 2, py5.height / 2)
    
    py5.text_font(italic_font)
    py5.text("Thank you for your participation", py5.width / 2, py5.height * 0.85)

# --- The Visualization Session (Implied Structure) ---

def key_pressed():
    """ Controls the progression of the experiment state. """
    global current_state
    
    if py5.key == ' ':
        # Linear progression logic for a single experiment session
        if current_state == STATE_WELCOME:
            current_state = STATE_RESTING_RECORDING
        
        elif current_state == STATE_RESTING_RECORDING:
            current_state = STATE_NEUROFEEDBACK_INSTRUCTIONS
        
        elif current_state == STATE_NEUROFEEDBACK_INSTRUCTIONS:
            current_state = STATE_END
    
    # Safety: Esc to exit full screen mode
    if py5.key == py5.ESC:
        py5.exit_sketch()

if __name__ == "__main__":
    py5.run_sketch()