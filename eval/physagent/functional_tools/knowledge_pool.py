'''
this is some necessary knowledge for PhysAgent
'''
attribute_table = {
    'brittle': {
        'most':
            [
                'cupcake', 'ice cream', 'shot glass', 'egg', 'chocolate cake', 'alcohol', 'birthday cake',
                'dinner plate', 'camera lens', 'cookie', 'hamburger', 'disk', 'cake', 'hand glass', 'yogurt',
                'cup', 'chinaware', 'eclair', 'soup bowl', 'beaker', 'aquarium', 'chocolate mousse'
            ],
        'least':
            [
                'handsaw', 'leather', 'shovel', 'clothespin', 'hammer', 'water bottle', 'colander', 'brace', 'poster',
                 'ashcan', 'necklace', 'bag', 'teddy', 'fork', 'cone', 'saucepan', 'strainer', 'tool', 'paintbrush',
                 'crowbar', 'bonnet', 'cigar box', 'barbell', 'cube', 'chopstick', 'dustpan', 'jacket', 'scarf',
                 'paperback book', 'ax', 'bowler hat', 'clutch bag', 'baseball cap', 'ballet skirt', 'basketball',
                 'coffeepot', 'gym shoe', 'dishtowel', 'oar', 'sponge', 'adhesive tape', 'ball', 'drawer', 'pencil',
                 'glycerol', 'tank top', 'duct tape', 'knob', 'cleat', 'map', 'apron', 'bulletin board', 'coaster',
                 'wrench', 'binder', 'plume', 'briefcase', 'baseball bat', 'hook', 'cooking utensil', 'handle',
                 'handcuff', 'volleyball', 'duffel bag', 'dog collar', 'trench coat', 'bubble gum', 'life buoy',
                 'quilt', 'envelope', 'crate', 'clothing', 'cymbal', 'shears', 'life jacket', 'rug', 'funnel', 'hinge',
                 'washcloth', 'tennis ball', 'wastepaper basket', 'wine bucket', 'grocery bag', 'latch', 'sombrero',
                 'eraser', 'football helmet', 'bulletproof vest', 'doorknob', 'condiment', 'lemonade', 'vest',
                 'thermos', 'sunhat', 'meat', 'ski boot', 'ribbon', 'broom', 'sword', 'blanket', 'coat', 'knee pad'
            ]
    },
    'stiff': {
        'least':
            [
                'wineglass', 'tomato', 'pickle', 'plant', 'pancake', 'crouton', 'chocolate cake', 'burrito', 'envelope',
                 'bouquet', 'raspberry', 'ring', 'wind chime', 'cigarette', 'grape', 'egg', 'cake', 'waffle', 'candy',
                 'cracker', 'candy bar', 'omelet', 'blueberry', 'pizza', 'cupcake', 'persimmon', 'cookie', 'kiwi',
                 'matchbox', 'chocolate mousse', 'green onion', 'quiche', 'gelatin', 'hamburger', 'pinwheel', 'cereal',
                 'bubble gum'
             ],
        'most':
            [
                'padlock', 'pillow', 'lego', 'fire alarm', 'hot plate', 'palette', 'sweat pants', 'coin',
                'screwdriver', 'drawer', 'checkerboard', 'keg', 'corkscrew', 'skewer', 'bucket', 'helmet',
                'cymbal', 'scraper', 'shoe', 'water faucet', 'sweater', 'recliner', 'wine bucket', 'liquor',
                'rug', 'boot', 'bow', 'file', 'knob', 'wrench', 'mailbox', 'block', 'dumbbell', 'faucet',
                'leather', 'scissors', 'life jacket', 'knee pad', 'scarf', 'ax', 'cube', 'life buoy',
                'paperweight', 'cylinder', 'shears', 'lawn mower', 'mallet', 'speaker', 'teakettle',
                'cabinet', 'shelf', 'baseball bat', 'frying pan', 'dish', 'chopping board'
            ]
    },
    'soft': {
        'least': [
            'blackboard', 'wind chime', 'oil lamp', 'pitchfork', 'coatrack', 'pole', 'remote control', 'dagger',
            'strainer', 'short pants', 'water pistol', 'bulletin board', 'pencil', 'aquarium', 'ax', 'dustpan',
            'golf club', 'card', 'lego', 'birdcage', 'knife', 'coaster', 'magnet', 'chinaware', 'wineglass',
            'drawer', 'projector', 'record player', 'laptop', 'computer keyboard', 'shears', 'coin', 'colander',
            'saucepan'
        ],
        'most': [
            'peach', 'burrito', 'ham', 'plume', 'flag', 'tarpaulin', 'headband', 'sock', 'pastry', 'bubble gum',
            'salami', 'blackberry', 'birthday cake', 'dishtowel', 'pretzel', 'beanie', 'sweatband', 'trouser',
            'handkerchief', 'bow tie', 'bread', 'milkshake', 'table mat', 'scarf', 'blazer', 'cloak', 'beret',
            'cowboy hat', 'sunflower', 'eclair', 'blanket', 'sweater', 'leather', 'kiwi', 'bulletproof vest',
            'pillow'
        ]
    },
    'smooth': {
        'most': [
            'magazine', 'memory device', 'chessboard', 'trouser', 'crayon', 'soya milk',
           'business card', 'hand glass', 'belt buckle', 'balloon', 'hardback', 'beer bottle',
           'knob', 'disk', 'spear', 'bow tie', 'shampoo', 'rearview mirror', 'honey', 'gift wrap',
           'tank top', 'aerosol', 'cymbal', 'shot glass', 'display', 'fire extinguisher', 'clothing',
           'stool', 'wine bottle', 'peanut butter', 'leather', 'watermelon', 'tape', 'cowboy hat',
           'jam', 'medicine', 'egg', 'steak knife', 'board', 'mirror', 'flannel', 'chinaware',
           'tray', 'soup bowl', 'calendar', 'sweat pants', 'wrench', 'banana', 'pajama', 'green onion',
           'jar', 'shower curtain'
        ],
        'least': [
            'trunk', 'bandage', 'brownie', 'wheel', 'cake', 'corkscrew', 'dog collar', 'almond', 'wreath',
            'avocado', 'lemon', 'pineapple', 'clementine', 'washcloth', 'cantaloupe', 'sweet potato', 'bagel',
            'beef', 'corn', 'cauliflower', 'keg', 'burrito', 'brussels sprouts', 'flower arrangement',
            'quesadilla'
        ]
    },
    'malleable': {
        'most': [
            'dollar', 'plastic bag', 'handkerchief', 'scarf', 'towel', 'dishtowel', 'ribbon', 'cardigan', 'bandage',
            'gift wrap', 'jacket', 'toilet tissue', 'tablecloth', 'orange juice', 'flag', 'shirt', 'pajama', 'leather',
            'kimono', 'liquor', 'skirt', 'clothing', 'washcloth', 'blanket', 'diaper', 'coat', 'trench coat'
        ],
        'least': [
            'flashlight', 'stool', 'remote control', 'water jug', 'bolt', 'turnip', 'bunk bed', 'baseball',
            'hard disc', 'television receiver', 'eggplant', 'earring', 'racket', 'dish', 'firework', 'cup',
            'chocolate bar', 'lantern', 'goggles', 'pepper mill', 'life buoy', 'mug', 'crutch', 'coffee maker',
            'magnet', 'cracker', 'footstool', 'memory device', 'lamp', 'loudspeaker', 'kitchen utensil', 'softball',
            'wine bucket', 'wineglass', 'rifle', 'crowbar', 'dustpan', 'kitchen sink', 'stick', 'martini', 'dumbbell',
            'ashtray'
        ]
    },
    'sharp': {
        'least': [
            'washbasin', 'footwear', 'lamb chop', 'coin', 'basketball', 'olive oil', 'timer', 'figurine', 'washcloth',
            'sweater', 'soap', 'shelf', 'bagel', 'calculator', 'sunhat', 'cupcake', 'band aid', 'cauliflower',
            'hockey stick', 'quesadilla', 'urn', 'football helmet', 'mushroom', 'beanbag', 'baseball glove',
            'clutch bag', 'goggles', 'cracker', 'shampoo', 'dietary supplement', 'gelatin', 'apricot', 'planter',
            'toothbrush', 'jar', 'soccer ball', 'candy', 'shoe', 'basket', 'keycard', 'water faucet', 'grape',
            'lemonade', 'egg', 'teakettle', 'pizza', 'towel', 'eclair', 'liquor', 'bandage', 'persimmon', 'sunflower',
            'chocolate cake', 'coffee maker', 'pastry', 'lampshade', 'quiche', 'dinner plate', 'memory device',
            'camera', 'wok', 'bowl', 'stool', 'toilet tissue', 'cleaning implement'
        ],
        'most': [
            'handsaw', 'dagger', 'thumbtack', 'spear', 'shovel', 'scissors', 'hook', 'drill', 'shears', 'steak knife',
            'fork', 'grater', 'screwdriver', 'sword', 'syringe', 'safety pin', 'pocketknife', 'safety', 'pitchfork',
            'ax', 'knife', 'corkscrew'
        ]
    },
    'elastic': {
        'least': [
            'pitchfork', 'coaster', 'garlic', 'liquor', 'hammer', 'birdcage', 'jewelry', 'laptop', 'soup bowl',
            'nutcracker', 'funnel', 'cracker'
            ],
        'most': [
            'bubble gum', 'backpack', 'cylinder', 'garden hose', 'bow tie', 'rubber band', 'futon',
            'flower arrangement', 'balloon', 'ballet skirt', 'wig', 'ball', 'underwear', 'bowler hat',
            'wet suit', 'beach ball', 'kimono', 'tarpaulin', 'trench coat', 'blanket', 'shower curtain', 'dishrag'
        ]
    }
}
