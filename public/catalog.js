/* Domain catalogs. Each domain has 4 categories × 6 items; the 6th item in each
 * category is a new release with no interaction history. All titles are invented. */
(function (root) {
  const CATALOG = {
    movies: {
      label: 'Movie streaming', unit: 'titles', verb: 'watched',
      cats: ['Sci-fi', 'Romance', 'Thriller', 'Animation'],
      items: [
        ['Orbit of Ash', 'Low Tide Signal', 'The Ninth Colony', 'Glass Meridian', 'Static Bloom', 'Hollow Sun'],
        ['Letters to Lisbon', 'Monsoon Hours', 'Second Spring', 'The Paper Bridge', 'Late Trains', 'Saffron Summer'],
        ['Cold Ledger', 'The Quiet Floor', 'Nine Below', 'Dead Reckoning Road', 'Blind Corner', 'Red Harbor'],
        ['Pip and the Lantern', 'Cloudhoppers', 'The Brass Fox', 'Moss Kingdom', 'Tiny Thunder', 'Kite Street'],
      ],
    },
    music: {
      label: 'Music platform', unit: 'tracks', verb: 'played',
      cats: ['Lo-fi', 'Indie rock', 'Hip-hop', 'Classical'],
      items: [
        ['Rain on Tin', 'Desk Lamp', 'Slow Bus', '3 AM Notes', 'Window Seat', 'Soft Static'],
        ['Parking Lot Anthem', 'Velvet Fuse', 'Northbound', 'Cheap Sunglasses', 'Paper Planes Club', 'Neon Garage'],
        ['Concrete Halo', 'Block Theory', 'Gold Teeth Gospel', 'Metro Lines', 'Heavy Crown', 'Night Shift Flow'],
        ['Nocturne in Grey', 'Cello for Two', 'The Winter Fugue', 'Adagio Blue', 'Harbor Sonata', 'Prelude No. 9'],
      ],
    },
    shop: {
      label: 'E-commerce', unit: 'products', verb: 'bought',
      cats: ['Running', 'Kitchen', 'Gaming', 'Skincare'],
      items: [
        ['Trail Runner 4', 'Wicking Tee', 'Hydration Vest', 'Carbon Insoles', 'Night Reflect Band', 'Stride Pro Socks'],
        ['Cast Iron Skillet', 'Pour-over Kettle', 'Knife Roll Set', 'Spice Tin Rack', 'Silicone Lids', 'Smart Thermometer'],
        ['Mech Keyboard 75%', 'Wireless Pad X', 'Desk Mat XL', 'Clicky Mouse', 'Stream Light', 'Haptic Headset'],
        ['Barrier Cream', 'Vitamin C Serum', 'Gel Cleanser', 'SPF 50 Fluid', 'Overnight Mask', 'Peptide Toner'],
      ],
    },
    courses: {
      label: 'Online education', unit: 'courses', verb: 'enrolled in',
      cats: ['Machine learning', 'Design', 'Finance', 'Languages'],
      items: [
        ['Intro to Neural Nets', 'Practical Pandas', 'Recommender Systems 101', 'Bayesian Thinking', 'Vision Transformers', 'GANs from Scratch'],
        ['Type for Screens', 'Figma Systems', 'Colour Theory Lab', 'Motion Basics', 'UX Research Sprint', 'Brand Identity'],
        ['Personal Finance', 'Reading a Balance Sheet', 'Options Explained', 'Startup Valuation', 'Tax Basics', 'Index Investing'],
        ['Spanish in 30 Days', 'Japanese Kana', 'Business English', 'French Phonetics', 'Hindi Conversation', 'German Grammar'],
      ],
    },
    jobs: {
      label: 'Job portal', unit: 'listings', verb: 'applied to',
      cats: ['Engineering', 'Design', 'Marketing', 'Data'],
      items: [
        ['Backend Engineer', 'Mobile Developer', 'SRE, Platform', 'Frontend Engineer', 'Embedded Firmware', 'ML Infra Engineer'],
        ['Product Designer', 'Visual Designer', 'UX Researcher', 'Design Systems Lead', 'Motion Designer', 'Service Designer'],
        ['Growth Marketer', 'Content Strategist', 'SEO Specialist', 'Brand Manager', 'Lifecycle Marketer', 'Community Lead'],
        ['Data Analyst', 'Analytics Engineer', 'Data Scientist', 'BI Developer', 'Data Platform Eng.', 'Applied Scientist'],
      ],
    },
    food: {
      label: 'Food delivery', unit: 'dishes', verb: 'ordered',
      cats: ['North Indian', 'East Asian', 'Healthy bowls', 'Desserts'],
      items: [
        ['Dal Makhani', 'Paneer Tikka', 'Butter Naan Box', 'Chole Bhature', 'Hyderabadi Biryani', 'Tandoori Platter'],
        ['Chilli Garlic Noodles', 'Bao Trio', 'Tom Yum Soup', 'Sushi Starter Set', 'Kimchi Fried Rice', 'Ramen Bowl'],
        ['Quinoa Power Bowl', 'Greek Salad', 'Tofu Poke', 'Millet Khichdi', 'Falafel Wrap', 'Avocado Toast'],
        ['Gulab Jamun', 'Molten Lava Cake', 'Mango Kulfi', 'Tiramisu Cup', 'Rasmalai', 'Biscoff Cheesecake'],
      ],
    },
    news: {
      label: 'News', unit: 'stories', verb: 'read',
      cats: ['Tech', 'Politics', 'Sport', 'Science'],
      items: [
        ['Chip Supply Outlook', 'Inside a Startup Layoff', 'The Battery Race', 'Open Models Explained', 'Phone Repair Rules', 'AI in Classrooms'],
        ['Budget Session Recap', 'State Polls Preview', 'Trade Deal Talks', 'Policy Explainer', 'Election Data Desk', 'Monsoon Session'],
        ['Test Series Review', 'Transfer Window', 'Marathon Season', 'Kabaddi League Final', 'Chess Olympiad', 'Hockey Comeback'],
        ['Deep Sea Survey', 'Exoplanet Atlas', 'Monsoon Models', 'Gene Editing Update', 'Fusion Milestone', 'Moon Dust Study'],
      ],
    },
  };
  root.CATALOG = CATALOG;
})(typeof window !== 'undefined' ? window : globalThis);
