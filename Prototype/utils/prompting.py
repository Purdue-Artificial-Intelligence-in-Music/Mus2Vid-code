

def get_prompt(comp_year):
    prompt = ''
    if (comp_year < 1400):
        prompt = 'A painting in the style of Giotto di Bondone of a musician in an italian town square'
    elif (comp_year < 1600):
        prompt = 'A painting in the style of Leonardo Da Vinci of a lute player performing for the royal family'
    elif (comp_year < 1750):
        prompt = 'A painting in the style of Johannes Vermeer of the girl with a pearl earing playing piano'
    elif (comp_year < 1830):
        prompt = 'A painting in the style of Jacque-Louis David of two knights dueling in the French country'
    elif (comp_year < 1920):
        prompt = 'A painting in the style of Claude Monet of a peaceful koi pond in a park in Argentina'
    else:
        prompt = 'A painting in the style of Andy Worhol of a cello player on stage'

    return(prompt)