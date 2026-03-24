// Configuration file for API keys and endpoints
const CONFIG = {
    // Backend API endpoint
    API_ENDPOINT: 'http://localhost:5000/api/classify',
    
    // Image search API keys
    UNSPLASH_API_KEY: 'K0Z0oC5jL19HkMEiVJFEkcaGUI5QgJJdvmL01ZdJBjE', // Your Unsplash Access Key
    FLICKR_API_KEY: 'YOUR_FLICKR_API_KEY',     // Get from https://www.flickr.com/services/api/
    
    // Fallback bird images (British species)
    BIRD_IMAGES: {
        'Barn Owl': 'https://upload.wikimedia.org/wikipedia/commons/thumb/c/c6/Tyto_alba_-British_Wildlife_Centre%2C_Surrey%2C_England-8a_%281%29.jpg/800px-Tyto_alba_-British_Wildlife_Centre%2C_Surrey%2C_England-8a_%281%29.jpg',
        'European Robin': 'https://upload.wikimedia.org/wikipedia/commons/thumb/b/b8/Erithacus_rubecula_with_cocked_head.jpg/800px-Erithacus_rubecula_with_cocked_head.jpg',
        'House Sparrow': 'https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Passer_domesticus_male_%2815%29.jpg/800px-Passer_domesticus_male_%2815%29.jpg',
        'Common Blackbird': 'https://upload.wikimedia.org/wikipedia/commons/thumb/a/a9/Common_Blackbird.jpg/800px-Common_Blackbird.jpg',
        'Chaffinch': 'https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Chaffinch_%28Fringilla_coelebs%29.jpg/800px-Chaffinch_%28Fringilla_coelebs%29.jpg',
        'Great Spotted Woodpecker': 'https://upload.wikimedia.org/wikipedia/commons/thumb/9/9e/Great_spotted_woodpecker_%28Dendrocopos_major%29_female.jpg/800px-Great_spotted_woodpecker_%28Dendrocopos_major%29_female.jpg',
        'Eurasian Magpie': 'https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/Pica_pica_-_Compans_Caffarelli_-_2012-03-16.jpg/800px-Pica_pica_-_Compans_Caffarelli_-_2012-03-16.jpg',
        'Mallard': 'https://upload.wikimedia.org/wikipedia/commons/thumb/b/bf/Anas_platyrhynchos_male_female_quadrat.jpg/800px-Anas_platyrhynchos_male_female_quadrat.jpg',
        'Grey Heron': 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/35/Ardea_cinerea_3_%28Marek_Szczepanek%29.jpg/800px-Ardea_cinerea_3_%28Marek_Szczepanek%29.jpg',
        'Kingfisher': 'https://upload.wikimedia.org/wikipedia/commons/thumb/9/92/Common_Kingfisher_Alcedo_atthis.jpg/800px-Common_Kingfisher_Alcedo_atthis.jpg',
        'Starling': 'https://upload.wikimedia.org/wikipedia/commons/thumb/7/7d/Starling_%28Sturnus_vulgaris%29_-_geograph.org.uk_-_563935.jpg/800px-Starling_%28Sturnus_vulgaris%29_-_geograph.org.uk_-_563935.jpg',
        'Tawny Owl': 'https://upload.wikimedia.org/wikipedia/commons/thumb/5/54/Strix_aluco_3_%28Martin_Mecnarowski%29.jpg/800px-Strix_aluco_3_%28Martin_Mecnarowski%29.jpg'
    }
};

// Export for use in other files
if (typeof module !== 'undefined' && module.exports) {
    module.exports = CONFIG;
}