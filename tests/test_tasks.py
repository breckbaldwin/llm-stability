import os
import sys
sys.path.append(os.getcwd())
import tasks.professional_accounting
import tasks.navigate
import tasks.ruin_names
import tasks.geometric_shapes
import tasks.high_school_european_history
import tasks.college_mathematics

import pandas as pd
import pytest

def test_college_mathematics():
    config = {'prompt_type': 'v2',
              'shots': 0}
    rubrics = tasks.college_mathematics.get_test_data(config)
    print(rubrics[2])
    task_config = {'prompt_type': 'v2',
                   'shots': 0}
    row = pd.Series({'gt': '(A)', 'response': '{"Answer": "A"}'})
    assert tasks.college_mathematics.correct_fn(row, task_config) == True
    row = pd.Series({'gt': '(A)', 'response': '{"Answer": "B"}'})
    assert tasks.college_mathematics.correct_fn(row, task_config) == False
    row = pd.Series({'gt': '(A)', 'response': 'The answer is (A)'})
    assert tasks.college_mathematics.correct_fn(row, task_config) == True
    long_response = '''
    {
  "$schema": "http://json-schema.org/draft-04/schema#",
  "type": "object",
  "properties": {
    "Answer": {
      "type": "string",
      "enum" : ["A", "B", "C", "D"]
    }
  },
  "required": [
    "Answer"
  ]
}

{
  "Answer": "B"
}
''' #not valid json but recoverable by string processing
    row = pd.Series({'gt': '(B)', 'response': long_response})
    assert tasks.college_mathematics.correct_fn(row, task_config) == True

    deep_seek_response='''
<think>
First, I need to find where the linear transformation T maps the point (2, 1). I'm given that T maps (1, 2) to (2, 3) and (-1, 2) to (2, -3). Since T is linear, I can use the properties of linear transformations to solve this.

A linear transformation satisfies two key properties:

1. T(u + v) = T(u) + T(v) for any vectors u and v.

2. T(cu) = cT(u) for any scalar c and vector u.

I need to express (2, 1) as a linear combination of the given points (1, 2) and (-1, 2), because I know where those points are mapped to.

Let me denote the vectors:

Let u = (1, 2) and v = (-1, 2).

I know T(u) = (2, 3) and T(v) = (2, -3).

Now, I need to find T(2, 1). So, I should find scalars a and b such that:

a * u + b * v = (2, 1)

That is:

a * (1, 2) + b * (-1, 2) = (2, 1)

This gives me the system of equations:

For the x-component: a * 1 + b * (-1) = 2 ⇒ a - b = 2

For the y-component: a * 2 + b * 2 = 1 ⇒ 2a + 2b = 1

Simplify the second equation: 2a + 2b = 1 ⇒ divide by 2: a + b = 1/2

Now I have:

Equation 1: a - b = 2

Equation 2: a + b = 1/2

I can solve this system.

Add the two equations: (a - b) + (a + b) = 2 + 1/2 ⇒ 2a = 2.5 ⇒ 2a = 5/2 ⇒ a = 5/4

2 + 1/2 = 4/2 + 1/2 = 5/2, so 2a = 5/2, thus a = 5/4.

Now, from equation 2: a + b = 1/2

5/4 + b = 1/2

b = 1/2 - 5/4 = 2/4 - 5/4 = -3/4

So, b = -3/4

Now, since T is linear, T(a*u + b*v) = a*T(u) + b*T(v)

Therefore, T(2,1) = T(a*u + b*v) = a*T(u) + b*T(v) = a*(2,3) + b*(2,-3)

Plug in a = 5/4 and b = -3/4

So, a*T(u) = (5/4) * (2,3) = (5/4 * 2, 5/4 * 3) = (10/4, 15/4) = (5/2, 15/4)

10/4 simplifies to 5/2, and 15/4 is already simplified.

b*T(v) = (-3/4) * (2, -3) = (-3/4 * 2, -3/4 * (-3)) = (-6/4, 9/4) = (-3/2, 9/4)

-6/4 is -3/2, and 9/4 is fine.

Now, add them together:

T(2,1) = a*T(u) + b*T(v) = (5/2, 15/4) + (-3/2, 9/4)

Add x-components: 5/2 + (-3/2) = (5 - 3)/2 = 2/2 = 1

Add y-components: 15/4 + 9/4 = 24/4 = 6

So, T(2,1) = (1, 6)

Now, looking at the options:

(A) (1, 6)

(B) (-1, 4)

(C) (3, 2)

(D) (-4, 3)

So, (1,6) is option A.

I should double-check if I did everything correctly.

I expressed (2,1) as a linear combination of (1,2) and (-1,2).

I had a - b = 2 and a + b = 1/2.

a + b = 1/2? Let me verify that.

From the y-component: 2a + 2b = 1, so dividing by 2, a + b = 1/2, yes.

Then a - b = 2

Adding: 2a = 2 + 1/2 = 5/2, a = 5/4

Then b = 1/2 - a = 1/2 - 5/4 = 2/4 - 5/4 = -3/4, correct.

Then T(u) = (2,3), T(v) = (2,-3)

a*T(u) = (5/4)(2,3) = (10/4, 15/4) = (5/2, 15/4)

b*T(v) = (-3/4)(2,-3) = (-6/4, 9/4) = (-3/2, 9/4)

Now, (5/2, 15/4) + (-3/2, 9/4) = (5/2 - 3/2, 15/4 + 9/4) = (2/2, 24/4) = (1, 6)

Yes.

Since T is linear, I can find the matrix representation.

Let me find the matrix of T.

Assume the standard basis. Let e1 = (1,0), e2 = (0,1).

But I don't have T on the standard basis directly. I have T on (1,2) and (-1,2).

(1,2) and (-1,2) are two vectors.

I can express the standard basis in terms of these or find the transformation.

Set up the matrix.

Suppose T is given by a matrix A = [a b; c d], so that T(x,y) = (ax + by, cx + dy)

Now, T(1,2) = (a*1 + b*2, c*1 + d*2) = (a + 2b, c + 2d) = (2,3)

So, a + 2b = 2

c + 2d = 3

Similarly, T(-1,2) = (a*(-1) + b*2, c*(-1) + d*2) = (-a + 2b, -c + 2d) = (2, -3)

So, -a + 2b = 2

And -c + 2d = -3

Now I have two equations for a and b, and two for c and d.

First, for a and b:

Eq1: a + 2b = 2

Eq2: -a + 2b = 2

Add Eq1 and Eq2: (a + 2b) + (-a + 2b) = 2 + 2 => 4b = 4 => b = 1

Then from Eq1: a + 2(1) = 2 => a + 2 = 2 => a = 0

a = 0?

Now, for c and d:

Eq3: c + 2d = 3

Eq4: -c + 2d = -3

Add Eq3 and Eq4: (c + 2d) + (-c + 2d) = 3 + (-3) => 4d = 0 => d = 0

Then from Eq3: c + 2(0) = 3 => c = 3

But from Eq4: -c + 2(0) = -3 => -c = -3 => c = 3, consistent.

But d = 0.

Now, T(2,1) = A * (2,1) = [a b; c d] * (2,1) = [0 1; 3 0] * (2,1) = (0*2 + 1*1, 3*2 + 0*1) = (1, 6)

Same as before.

In the matrix, a=0, b=1, c=3, d=0.

T(1,2) = (0*1 + 1*2, 3*1 + 0*2) = (2,3), good.

T(-1,2) = (0*(-1) + 1*2, 3*(-1) + 0*2) = (2, -3), good.

T(2,1) = (0*2 + 1*1, 3*2 + 0*1) = (1,6), perfect.

So, it maps to (1,6), which is option A.

In the options, (A) is (1,6), so
'''
    row = pd.Series({'gt': '(A)', 'response': deep_seek_response})
    assert tasks.college_mathematics.correct_fn(row, task_config) == True



def test_high_school_european_history():
    config = {'prompt_type': 'v2',
              'shots': 0}
    rubrics = tasks.high_school_european_history.get_test_data(config)
    assert rubrics[0] == {'input': 'This question refers to the following information.\nIn order to make the title of this discourse generally intelligible, I have translated the term "Protoplasm," which is the scientific name of the substance of which I am about to speak, by the words "the physical basis of life." I suppose that, to many, the idea that there is such a thing as a physical basis, or matter, of life may be novel—so widely spread is the conception of life as something which works through matter. … Thus the matter of life, so far as we know it (and we have no right to speculate on any other), breaks up, in consequence of that continual death which is the condition of its manifesting vitality, into carbonic acid, water, and nitrogenous compounds, which certainly possess no properties but those of ordinary matter.\nThomas Henry Huxley, "The Physical Basis of Life," 1868\nFrom the passage, one may infer that Huxley argued that "life" was\n(A) a force that works through matter. (B) essentially a philosophical notion. (C) merely a property of a certain kind of matter. (D) a supernatural phenomenon. ', 'target': '(C)'}
    
    config = {'prompt_type': 'v2',
              'shots': 'few'}

    rubrics = tasks.high_school_european_history.get_test_data(config)
    assert rubrics[0] == {'input': 'This question refers to the following information.\nAlbeit the king\'s Majesty justly and rightfully is and ought to be the supreme head of the Church of England, and so is recognized by the clergy of this realm in their convocations, yet nevertheless, for corroboration and confirmation thereof, and for increase of virtue in Christ\'s religion within this realm of England, and to repress and extirpate all errors, heresies, and other enormities and abuses heretofore used in the same, be it enacted, by authority of this present Parliament, that the king, our sovereign lord, his heirs and successors, kings of this realm, shall be taken, accepted, and reputed the only supreme head in earth of the Church of England, called Anglicans Ecclesia; and shall have and enjoy, annexed and united to the imperial crown of this realm, as well the title and style thereof, as all honors, dignities, preeminences, jurisdictions, privileges, authorities, immunities, profits, and commodities to the said dignity of the supreme head of the same Church belonging and appertaining; and that our said sovereign lord, his heirs and successors, kings of this realm, shall have full power and authority from time to time to visit, repress, redress, record, order, correct, restrain, and amend all such errors, heresies, abuses, offenses, contempts, and enormities, whatsoever they be, which by any manner of spiritual authority or jurisdiction ought or may lawfully be reformed, repressed, ordered, redressed, corrected, restrained, or amended, most to the pleasure of Almighty God, the increase of virtue in Christ\'s religion, and for the conservation of the peace, unity, and tranquility of this realm; any usage, foreign land, foreign authority, prescription, or any other thing or things to the contrary hereof notwithstanding.\nEnglish Parliament, Act of Supremacy, 1534\nFrom the passage, one may infer that the English Parliament wished to argue that the Act of Supremacy would\n(A) give the English king a new position of authority (B) give the position of head of the Church of England to Henry VIII alone and exclude his heirs (C) establish Calvinism as the one true theology in England (D) end various forms of corruption plaguing the Church in England A:The answer is (D).\nThis question refers to the following information.\nRead the following excerpt.\nThe revolutionary seed had penetrated into every country and spread more or less. It was greatly developed under the régime of the military despotism of Bonaparte. His conquests displaced a number of laws, institutions, and customs; broke through bonds sacred among all nations, strong enough to resist time itself; which is more than can be said of certain benefits conferred by these innovators.\nThe monarchs will fulfil the duties imposed upon them by Him who, by entrusting them with power, has charged them to watch over the maintenance of justice, and the rights of all, to avoid the paths of error, and tread firmly in the way of truth. Placed beyond the passions which agitate society, it is in days of trial chiefly that they are called upon to despoil realities of their false appearances, and to show themselves as they are, fathers invested with the authority belonging by right to the heads of families, to prove that, in days of mourning, they know how to be just, wise, and therefore strong, and that they will not abandon the people whom they ought to govern to be the sport of factions, to error and its consequences, which must involve the loss of society.\nUnion between the monarchs is the basis of the policy which must now be followed to save society from total ruin. . . .\nLet them not confound concessions made to parties with the good they ought to do for their people, in modifying, according to their recognized needs, such branches of the administration as require it.\nLet them be just, but strong; beneficent, but strict.\nLet them maintain religious principles in all their purity, and not allow the faith to be attacked and morality interpreted according to the social contract or the visions of foolish sectarians.\nLet them suppress Secret Societies; that gangrene of society.\n—Klemens von Metternich, Political Confession of Faith, 1820\nWhich of the following was the greatest cause of the fears expressed by Metternich in the document above?\n(A) The ideas of personal liberty and nationalism conceived during the Enlightenment resulted in radical revolutions that could spread throughout Europe. (B) The conquest of Europe by Napoleon led to the creation of new factions and shifted the European balance of power. (C) The power of monarchs had grown to the point where it needed to be checked by other powers within each nation or domination of civilians would occur. (D) The rising and falling economic cycle of the newly emerging capitalist economy could lead to civilian unrest that must be suppressed. A:The answer is (A).\nThis question refers to the following information.\nThe excerpts below are from the Navigation Acts of 1651.\n[A]fter the first day of December, one thousand six hundred fifty and one, and from thence forwards, no goods or commodities whatsoever of the growth, production or manufacture of Asia, Africa or America, or of any part thereof; or of any islands belonging to them, or which are described or laid down in the usual maps or cards of those places, as well of the English plantations as others, shall be imported or brought into this Commonwealth of England, or into Ireland, or any other lands, islands, plantations, or territories to this Commonwealth belonging, or in their possession, in any other ship or ships, vessel or vessels whatsoever, but only in such as do truly and without fraud belong only to the people of this Commonwealth, or the plantations thereof, as the proprietors or right owners thereof; and whereof the master and mariners are also of the people of this Commonwealth, under the penalty of the forfeiture and loss of all the goods that shall be imported contrary to this act, , , ,\n[N]o goods or commodities of the growth, production, or manufacture of Europe, or of any part thereof, shall after the first day of December, one thousand six hundred fifty and one, be imported or brought into this Commonwealth of England, or any other lands or territories to this Commonwealth belonging, or in their possession, in any ship or ships, vessel or vessels whatsoever, but in such as do truly and without fraud belong only to the people of this Commonwealth, and in no other, except only such foreign ships and vessels as do truly and properly belong to the people of that country or place, of which the said goods are the growth, production or manufacture.\nWhich of the following best describes the outcome of the Navigation Acts of 1651?\n(A) They served as a catalyst for the growth of English shipping and overseas trade, but did little to limit the prospects of the Dutch in the seventeenth century. (B) They brought about almost immediate hardships for the Dutch economy as their dominance of overseas trade quickly ended. (C) They were rescinded during the restoration of the Stuarts as they sought normal diplomatic relations with the Dutch so not as to need Parliament\'s financial support for war. (D) They led to nearly a century of recurrent war between England and the Netherlands, which would not end until after American independence. A:The answer is (A).\nThis question refers to the following information.\nIn Russia there was nothing going on well, and [Souvarine] was in despair over the news he had received. His old companions were all turning to the politicians; the famous Nihilists who made Europe tremble-sons of village priests, of the lower middle class, of tradesmen-could not rise above the idea of national liberation, and seemed to believe that the world would be delivered-when they had killed their despot&…\n"Foolery! They\'ll never get out of it with their foolery."\nThen, lowering his voice still more, in a few bitter words he described his old dream of fraternity. He had renounced his rank and his fortune; he had gone among workmen, only in the hope of seeing at last the foundation of a new society of labour in common. All the sous in his pockets had long gone to the urchins of the settlement; he had been as tender as a brother with the colliers, smiling at their suspicion, winning them over by his quiet workmanlike ways and his dislike of chattering. But decidedly the fusion had not taken place.\nHis voice changed, his eyes grew bright, he fixed them on étienne, directly addressing him:\n"Now, do you understand that? These hatworkers at Marseilles who have won the great lottery prize of a hundred thousand francs have gone off at once and invested it, declaring that they are going to live without doing anything! Yes, that is your idea, all of you French workmen; you want to unearth a treasure in order to devour it alone afterwards in some lazy, selfish corner. You may cry out as much as you like against the rich, you haven\'t got courage enough to give back to the poor the money that luck brings you. You will never be worthy of happiness as long as you own anything, and your hatred of the bourgeois proceeds solely from an angry desire to be bourgeois yourselves in their place."\némile Zola, French writer, Germinal, 1885\nThe passage displays the direct concern for the welfare of the working classes that was typically a part of which movement?\n(A) Capitalist (B) Scientific (C) Communist (D) Existentialist A:The answer is (C).\nThis question refers to the following information.\nThe following excerpt is from a pamphlet.\nYou will do me the justice to remember, that I have always strenuously supported the Right of every man to his own opinion, however different that opinion might be to mine. He who denies to another this right, makes a slave of himself to his present opinion, because he precludes himself the right of changing it.\nThe most formidable weapon against errors of every kind is Reason. I have never used any other, and I trust I never shall.\nThe circumstance that has now taken place in France of the total abolition of the whole national order of priesthood, and of everything appertaining to compulsive systems of religion, and compulsive articles of faith, has not only precipitated my intention, but rendered a work of this kind exceedingly necessary, lest in the general wreck of superstition, of false systems of government, and false theology, we lose sight of morality, of humanity, and of the theology that is true.\nI believe in one God, and no more; and I hope for happiness beyond this life.\nI believe in the equality of man; and I believe that religious duties consist in doing justice, loving mercy, and endeavoring to make our fellow-creatures happy.\nI do not believe in the creed professed by the Jewish church, by the Roman church, by the Greek church, by the Turkish church, by the Protestant church, nor by any church that I know of. My own mind is my own church.\nAll national institutions of churches, whether Jewish, Christian or Turkish, appear to me no other than human inventions, set up to terrify and enslave mankind, and monopolize power and profit.\nI do not mean by this declaration to condemn those who believe otherwise; they have the same right to their belief as I have to mine.\n—Thomas Paine, The Age of Reason, 1794–1795\nWhich of the following Enlightenment philosophes designed a system of checks and balances for government to avoid abuses of power?\n(A) Jean Jacques Rousseau (B) Baron Montesquieu (C) Mary Wollstonecraft (D) Adam Smith A:The answer is (B). This question refers to the following information.\nIn order to make the title of this discourse generally intelligible, I have translated the term "Protoplasm," which is the scientific name of the substance of which I am about to speak, by the words "the physical basis of life." I suppose that, to many, the idea that there is such a thing as a physical basis, or matter, of life may be novel—so widely spread is the conception of life as something which works through matter. … Thus the matter of life, so far as we know it (and we have no right to speculate on any other), breaks up, in consequence of that continual death which is the condition of its manifesting vitality, into carbonic acid, water, and nitrogenous compounds, which certainly possess no properties but those of ordinary matter.\nThomas Henry Huxley, "The Physical Basis of Life," 1868\nFrom the passage, one may infer that Huxley argued that "life" was\n(A) a force that works through matter. (B) essentially a philosophical notion. (C) merely a property of a certain kind of matter. (D) a supernatural phenomenon. ', 'target': '(C)'}



def test_geometric_shapes():
    config = {'prompt_type': 'v2',
              'shots': 0}
    rubrics = tasks.geometric_shapes.get_test_data(config)
    assert rubrics[0] ==  {'input': 'This SVG path element <path d="M 55.57,80.69 L 57.38,65.80 M 57.38,65.80 L 48.90,57.46 M 48.90,57.46 L 45.58,47.78 M 45.58,47.78 L 53.25,36.07 L 66.29,48.90 L 78.69,61.09 L 55.57,80.69"/> draws a\nOptions:\n(A) circle\n(B) heptagon\n(C) hexagon\n(D) kite\n(E) line\n(F) octagon\n(G) pentagon\n(H) rectangle\n(I) sector\n(J) triangle', 'target': '(B)'}

def test_ruin_names():
    config = {'prompt_type': 'v2',
              'shots': 0}
    data = tasks.ruin_names.get_test_data(config)
    assert data[0]== {
        'input': "Which of the following is a humorous edit of this artist or movie name: 'rain man'?\nOptions:\n(A) ruin man\n(B) rains man\n(C) rain men\n(D) rainmman", 
        'target': '(A)'}
    config = {'prompt_type': 'v2',
              'shots': 'few'}
    data = tasks.ruin_names.get_test_data(config)
    assert data[0]['input'] == 'Which of the following is a humorous edit of this artist or movie name: \'whitesnake\'?\nOptions:\n(A) whitesnape\n(B) whitesnapke\n(C) whitesnuake\n(D) mwhitesnake A:Let\'s think step by step.\nThe original name is "whitesnake". This is the name of an old English hard rock band. It is a compound word, formed by the words "white" and "snake".\n(A) "whitesnape": It is formed by the combination of "white" and "snake"; therefore, "snake" has been changed to "snape". Snape makes a reference to the fictional character Severus Snape in the Harry Potter series, so (A) is indeed a meaningful and funny edit.\n(B) "whitesnapke": It is formed by the combination of "white" and "snapke", but "snapke" is not an actual word; therefore, "whitesnapke" is not humorous.\n(C) "whitesnuake": It is formed by the combination of "white" and "snuake", but "snuake" is not an actual word; therefore, "whitesnuake" is not humorous.\n(D) "mwhitesnake": It is formed by the combination of "m", "white", and "snake", but the prefix "-m "seems arbitrary; therefore, "mwhitesnake" is not meaningful or humorous.\nAbove the above, the only humorous edit is (A). So the answer is (A).\nWhich of the following is a humorous edit of this artist or movie name: \'one of our dinosaurs is missing\'?\nOptions:\n(A) ofne of our dinosaurs is missing\n(B) one af our dinosaurs is missing\n(C) one of our dinosaurs is pissing\n(D) one of our dinosaur is missing A:Let\'s think step by step.\nThe original name is "one of our dinosaurs is missing". This is the name of an old British movie.\n(A) "ofne of our dinosaurs is missing": Here "one of" is changed to "ofne", but the word "ofne" is not an actual word.\n(B) "one af our dinosaurs is missing": Here the word "of" is changed to "af", but the word "af" is not an actual word.\n(C) "one of our dinosaurs is pissing": Here the word "missing" is changed to "pissing", and "one of our dinosaurs is pissing" is indeed a very whimsical and mischievous edit. This change truly ruins the original title of the movie.\n(D) "one of our dinosaur is missing": Here the word "dinosaurs" is changed to "dinosaur", but "dinosaur" is singular but should be plural in the title; this change therefore feels arbitrary and not humorous.\nAbove the above, the only humorous edit is (C).\nAbove the above, the only humorous edit is (C). So the answer is (C).\nWhich of the following is a humorous edit of this artist or movie name: \'counting crows\'?\nOptions:\n(A) countingy crows\n(B) counting cows\n(C) courting crows\n(D) coutnting crows A:Let\'s think step by step.\nThe original name is "counting crows". This is the name of an American rock band. Historically, the band name comes from the British nursery rhyme "One for Sorrow", which is about counting of magpies.\n(A) "countingy crows": Here the word "counting" is changed to "countingy", but the word "countingy" is not an actual word.\n(B) "counting cows": Here the word "crows" is changed to "cows", and this is indeed a playful and meaningful edit that ruins the original name of the band.\n(C) "courting crows": Here the word "counting" is changed to "courting", and "courting" is an actual word; however, "courting crows" does not sound as humorous as "counting cows".\n(D) "coutnting crows": Here the word "counting" is changed to "coutnting", but the word "coutnting" is not an actual word.\nAbove the above, the only humorous edit is (B). So the answer is (B). Which of the following is a humorous edit of this artist or movie name: \'rain man\'?\nOptions:\n(A) ruin man\n(B) rains man\n(C) rain men\n(D) rainmman'


def test_professional_accounting():
    """
    For V1/V2 runs mmlu data was normalized to mirror the format for 
    lukaemon/bbh data.
    """
    data = \
        tasks.professional_accounting.get_test_data(
            config = {'prompt_type': 'minimal',
              'shots': 0}
        )
    assert data[0]['target'] == '164%' #index of answer
    assert data[0]['input']== "You bought a limousine for $98,000 and are planning to rent it for weddings, ceremonies and parties at $245 per hour. If you estimate the car will be hired for 2 hours a day on average, with daily costs at about $50, what is the estimated yearly yield on your investment if you work all year round, i.e. every day of the year, including any festivities and weekends?\n[164%, 1.64%, 0.45%, 183%]"
    
    config = {'prompt_type': 'v2',
              'shots': 0}
    data = tasks.professional_accounting.get_test_data(config)
    assert data[0]['input'] == 'You bought a limousine for $98,000 and are planning to rent it for weddings, ceremonies and parties at $245 per hour. If you estimate the car will be hired for 2 hours a day on average, with daily costs at about $50, what is the estimated yearly yield on your investment if you work all year round, i.e. every day of the year, including any festivities and weekends?\n(A) 164%. (B) 1.64%. (C) 0.45%. (D) 183%. '
    assert data[0]['target'] == '(A)'

    config = {'prompt_type': 'v2',
              'shots': 'few'}
    data = tasks.professional_accounting.get_test_data(config)
    assert data[0]['input'] == "An auditor traces the serial numbers on equipment to a nonissuer’s subledger. Which of the following management assertions is supported by this test?\n(A) Valuation and allocation (B) Completeness (C) Rights and obligations (D) Presentation and disclosure A:The answer is (B).\nOne hundred years ago, your great-great-grandmother invested $100 at 5% yearly interest. What is the investment worth today?\n(A) $13,000 (B) $600 (C) $15,000 (D) $28,000 A:The answer is (A).\nOn January 1, year 1, Alpha Co. signed an annual maintenance agreement with a software provider for $15,000 and the maintenance period begins on March 1, year 1. Alpha also incurred $5,000 of costs on January 1, year 1, related to software modification requests that will increase the functionality of the software. Alpha depreciates and amortizes its computer and software assets over five years using the straight-line method. What amount is the total expense that Alpha should recognize related to the maintenance agreement and the software modifications for the year ended December 31, year 1?\n(A) $5,000 (B) $13,500 (C) $16,000 (D) $20,000 A:The answer is (B).\nKrete is an unmarried taxpayer with income exclusively from wages. By December 31, year 1, Krete's employer has withheld $16,000 in federal income taxes and Krete has made no estimated tax payments. On April 15, year 2, Krete timely filed for an extension request to file her individual tax return, and paid $300 of additional taxes. Krete's year 1 tax liability was $16,500 when she timely filed her return on April 30, year 2, and paid the remaining tax liability balance. What amount would be subject to the penalty for underpayment of estimated taxes?\n(A) $0 (B) $500 (C) $1,650 (D) $16,500 A:The answer is (A).\nBox a nongovernmental not-for-profit organization had the following transactions during the year: Proceeds from sale of investments $80000 Purchase of property plant and equipment $10000 Proceeds from long-term debt $100000 Loss on sale of investment $5000 What amount should be reported as net cash provided by financing activities in Box's statement of cash flows?\n(A) $70,000 (B) $75,000 (C) $80,000 (D) 100000 A:The answer is (D). You bought a limousine for $98,000 and are planning to rent it for weddings, ceremonies and parties at $245 per hour. If you estimate the car will be hired for 2 hours a day on average, with daily costs at about $50, what is the estimated yearly yield on your investment if you work all year round, i.e. every day of the year, including any festivities and weekends?\n(A) 164%. (B) 1.64%. (C) 0.45%. (D) 183%. "
    assert data[0]['target'] == '(A)'

def test_answer_parsing():
    config = {'prompt_type': 'v2',
              'shots': 0}
    row = pd.Series({'response': 'The answer is (A)'})
    assert tasks.professional_accounting.answer_fn(row, config) == '(A)'

    row = pd.Series({'response': 'The answer is Yes'})
    assert tasks.navigate.answer_fn(row, config) == 'Yes'

    config = {'prompt_type': 'v2',
              'shots': 'few'}

    row = pd.Series({'response': 'The answer is (A)'})
    assert tasks.ruin_names.answer_fn(row, config) == '(A)'




def test_correct_fn():
    # professional_accounting
    config = {'prompt_type': 'v2',
              'shots': 0}
    row = pd.Series({'gt': '(A)', 'response': 'The answer is (A)'})
    assert tasks.professional_accounting.correct_fn(row, config)
    row = pd.Series({'gt': '(B)', 'response': 'The answer is (A)'})
    assert not tasks.professional_accounting.correct_fn(row, config)
    row = pd.Series({'gt': '(B)', 'response': 'The answer is ??'})
    assert tasks.professional_accounting.correct_fn(row, config) is None
    with pytest.raises(LookupError) as e:
        row = pd.Series({'response': 'The answer is (A), no (B)'})
        tasks.professional_accounting.correct_fn(row, config)
        assert e.value == "Blown UP: The answer is (A), no (B)"

    #navigate
    config = {'prompt_type': 'v2',
              'shots': 0}
    row = pd.Series({'gt': 'Yes', 'response': 'Yes'})
    assert tasks.navigate.correct_fn(row, config)
    row = pd.Series({'gt': 'No', 'response': 'Yes'})
    assert not tasks.navigate.correct_fn(row, config)
    row = pd.Series({'gt': 'Yes', 'response': 'Dunno'})
    assert tasks.navigate.correct_fn(row, config) is None
    with pytest.raises(LookupError) as e:
        row = pd.Series({'gt': 'Yes', 'response': 'Yes, or No'})
        tasks.navigate.correct_fn(row, config)
        assert e.value == "Blown UP: Yes, or No"





def test_logical_deduction():
    import tasks.logical_deduction
    config = {'prompt_type': 'v2',
              'shots': 0}
    data = tasks.logical_deduction.get_test_data(config)
    assert data[0]['input'] == 'The following paragraphs each describe a set of three objects arranged in a fixed order. The statements are logically consistent within each paragraph. On a branch, there are three birds: a blue jay, a quail, and a falcon. The falcon is to the right of the blue jay. The blue jay is to the right of the quail.\nOptions:\n(A) The blue jay is the second from the left\n(B) The quail is the second from the left\n(C) The falcon is the second from the left'
    assert data[0]['target'] == '(A)'

    config = {'prompt_type': 'v2',
              'shots': 'few'}
    data = tasks.logical_deduction.get_test_data(config)
    assert data[0]['input'] == '\nThe following paragraphs each describe a set of three objects arranged\nin a fixed order. The statements are logically consistent within each paragraph.\nIn a golf tournament, there were three golfers: Amy, Eli, and Eve. Eve finished\nabove Amy. Eli finished below Amy.\n\nOptions:\n\n(A) Amy finished last\n\n(B) Eli finished last\n\n(C) Eve finished last\'\ntarget: \'Let\'\'s think step by step.\n\n(1) Eve finished above Amy: "(above) ? Eve ? Amy ? (below)".\n\n(2) Eli finished below Amy: "(above) ? Amy ? Eli ? (below)".\n\n(3) Combining (1) and (2) we get the following ordering: "(above) Eve Amy Eli\n(below)".\n\nAccording to this ordering, the person who finished last (the one at the bottom\nof this list) is Eli.\n\nEli finished last. So the answer is (B).\n\nThe following paragraphs each describe a set of three objects arranged\n    in a fixed order. The statements are logically consistent within each paragraph.\n    On a shelf, there are three books: a white book, a green book, and an orange\n    book. The green book is to the right of the white book. The orange book is the\n    rightmost.\n\n    Options:\n\n    (A) The white book is the leftmost\n\n    (B) The green book is the leftmost\n\n    (C) The orange book is the leftmost\'\n    target: \'Let\'\'s think step by step.\n\n    (1) The green book is to the right of the white book: "(left) ? white ? green\n    ? (right)".\n\n    (2) The orange book is the rightmost: "(left) ? white ? green orange (right)".\n\n    (3) Combining (1) and (2) we get the following ordering: "(left) white green\n    orange (right)".\n\n    According to this ordering, the leftmost book is the white book.\n\n    The white book is the leftmost. So the answer is (A).\'\n\n\nThe following paragraphs each describe a set of three objects arranged\n    in a fixed order. The statements are logically consistent within each paragraph.\n    On a shelf, there are three books: a red book, a gray book, and a white book.\n    The white book is to the left of the gray book. The red book is the second from\n    the left.\n\n    Options:\n\n    (A) The red book is the leftmost\n\n    (B) The gray book is the leftmost\n\n    (C) The white book is the leftmost\'\n    target: \'Let\'\'s think step by step.\n\n    (1) The white book is to the left of the gray book: "(left) ? white ? gray ?\n    (right)".\n\n    (2) The red book is the second from the left: "(left) ? white red gray ? (right)".\n\n    (3) Combining (1) and (2) we get the following ordering: "(left) white red gray\n    (right)".\n\n    According to this ordering, the leftmost book is the white book.\n\n    The white book is the leftmost. So the answer is (C).\n The following paragraphs each describe a set of three objects arranged in a fixed order. The statements are logically consistent within each paragraph. On a branch, there are three birds: a blue jay, a quail, and a falcon. The falcon is to the right of the blue jay. The blue jay is to the right of the quail.\nOptions:\n(A) The blue jay is the second from the left\n(B) The quail is the second from the left\n(C) The falcon is the second from the left'
    print(data[0]['input'])
    assert data[0]['target'] == '(A)'



def test_navigate():
    """
    Format for lukaemon/bbh data. 
    """
    config = {'prompt_type': 'v2',
              'shots': 0}
    data = tasks.navigate.get_test_data(config)
    assert data[0]['input'] == 'If you follow these instructions, do you return to the starting point? Always face forward. Take 1 step backward. Take 9 steps left. Take 2 steps backward. Take 6 steps forward. Take 4 steps forward. Take 4 steps backward. Take 3 steps right.\nOptions:\n- Yes\n- No'
    assert data[0]['target'] == 'No'
    config = {'prompt_type': 'v2',
              'shots': 'few'}
    data = tasks.navigate.get_test_data(config)
    assert data[0]['input'] == "If you follow these instructions, do you return to the starting point? Turn left. Turn around. Turn left. Take 7 steps. Take 2 steps. Take 4 steps. Take 8 steps.\nOptions:\n- Yes\n- No A:Let's think step by step.\nWe start at the origin (0, 0), facing the positive y-axis.\n(1) Turn left: (0, 0), facing the negative x-axis.\n(2) Turn around: (0, 0), facing the positive x-axis.\n(3) Turn left: (0, 0), facing the positive y-axis.\n(4) Take 7 steps: (0, 7), facing the positive y-axis.\n(5) Take 2 steps: (0, 9), facing the positive y-axis.\n(6) Take 4 steps: (0, 13), facing the positive y-axis.\n(7) Take 8 steps: (0, 21), facing the positive y-axis.\nSince (0, 21) is not (0, 0), we are not where we started. So the answer is No.\nIf you follow these instructions, do you return to the starting point? Turn around. Take 1 step. Take 6 steps. Turn around. Take 6 steps. Take 9 steps. Take 1 step.\nOptions:\n- Yes\n- No A:Let's think step by step.\nWe start at the origin (0, 0), facing the positive y-axis.\n(1) Turn around: (0, 0), facing the negative y-axis.\n(2) Take 1 step: (0, -1), facing the negative y-axis.\n(3) Take 6 steps: (0, -7), facing the negative y-axis.\n(4) Turn around: (0, -7), facing the positive y-axis.\n(5) Take 6 steps: (0, -1), facing the positive y-axis.\n(6) Take 9 steps: (0, 8), facing the positive y-axis.\n(7) Take 1 step: (0, 9), facing the positive y-axis.\nSince (0, 9) is not (0, 0), we are not where we started. So the answer is No.\nIf you follow these instructions, do you return to the starting point? Always face forward. Take 2 steps right. Take 9 steps left. Take 7 steps right.\nOptions:\n- Yes\n- No A:Let's think step by step.\nWe start at the origin (0, 0), facing the positive y-axis.\n(1) Always face forward: (0, 0), facing the positive y-axis.\n(2) Take 2 steps right: (0, 2), facing the positive y-axis.\n(3) Take 9 steps left: (0, -7), facing the positive y-axis.\n(4) Take 7 steps right: (0, 7), facing the positive y-axis.\nSince (0, 0) is (0, 0), we are indeed where we started. So the answer is Yes. If you follow these instructions, do you return to the starting point? Always face forward. Take 1 step backward. Take 9 steps left. Take 2 steps backward. Take 6 steps forward. Take 4 steps forward. Take 4 steps backward. Take 3 steps right.\nOptions:\n- Yes\n- No"
    assert data[0]['target'] == 'No'

