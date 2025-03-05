def get_keyboard(width_keyboard, height_keyboard, offset_keyboard):
    """
    Draw a more spaced-out keyboard layout
    
    Args:
        width_keyboard (int): Total width of keyboard
        height_keyboard (int): Total height of keyboard
        offset_keyboard (tuple): Offset for keyboard placement
    
    Returns:
        list: Keyboard key points with reduced overlap
    """
    # Increase spacing between keys
    columns = 10
    rows = 5
    
    # Calculate key sizes with more padding
    box_width = width_keyboard / (columns + 1)
    box_height = height_keyboard / (rows + 1)
    
    # Create column and row positions with more spacing
    column = [int(i * width_keyboard / columns + offset_keyboard[0]) for i in range(columns)]
    row = [int(i * height_keyboard / rows + offset_keyboard[1]) for i in range(rows)]
    
    key_points = []
    keyboard_layout = [
        ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
        ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
        ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', ':'],
        ['Z', 'X', 'C', 'V', 'B', 'N', 'M', '.', "'", '?'],
        ['del', ' ', ' ', ' ', ' ', ' ', ' ', '!', '?', 'space']
    ]
    
    for r, row_keys in enumerate(keyboard_layout):
        for c, key_text in enumerate(row_keys):
            # Calculate center point
            center_x = column[c]
            center_y = row[r]
            
            # Calculate top-left and bottom-right with more spacing
            top_left = (int(center_x - box_width/2), int(center_y - box_height/2))
            bottom_right = (int(center_x + box_width/2), int(center_y + box_height/2))
            
            key_points.append([
                key_text, 
                (center_x, center_y),  # Center point
                top_left,              # Top-left corner
                bottom_right           # Bottom-right corner
            ])
    
    return key_points