

def get_prompt(subgenre):
    prompt = ''

    match subgenre:
        case 'Baroque':
            prompt = 'A painting in the style of Johannes Vermeer of the girl with a pearl earing playing piano'
        case 'Classical':
            prompt = 'A painting in the style of Jacque-Louis David of two knights dueling in the French country'
        case 'Romantic':
            prompt = 'A painting in the style of Claude Monet of a peaceful koi pond in a park in Argentina'
        case '20th Century':
            prompt = 'A painting in the style of Andy Worhol of a cello player on stage'

    return(prompt)