
class FringeDataProcessor:
    """Handles data scraping and preprocessing"""

    def __init__(self):
        self.dataset = []
        self.episode_id = 1

    def clean_text(self, text):
        """Clean and normalize text"""
        return ' '.join(text.strip().split())

    def extract_season_episode(self, page_text):
        """Extract season and episode numbers"""
        season = "Unknown"
        episode = "Unknown"

        match = re.search(r"Season\s+(\d+)[,\s]+Episode\s+(\d+)", page_text, re.IGNORECASE)
        if match:
            season = match.group(1)
            episode = match.group(2)
        else:
            match = re.search(r"S(\d+)E(\d+)", page_text, re.IGNORECASE)
            if match:
                season = match.group(1)
                episode = match.group(2)

        return season, episode

    def extract_writers_directors(self, page_text):
        """Extract writers and directors information"""
        writers = "Unknown"
        directors = "Unknown"

        writer_match = re.search(r'Written by[:\s]+([^\n]+)', page_text, re.IGNORECASE)
        director_match = re.search(r'Directed by[:\s]+([^\n]+)', page_text, re.IGNORECASE)

        if writer_match:
            writers = self.clean_text(writer_match.group(1))
        if director_match:
            directors = self.clean_text(director_match.group(1))

        return writers, directors

    def extract_cast(self, page_text):
        """Extract cast information"""
        cast_list = []
        matches = re.findall(r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s+as\s+([A-Z][a-zA-Z\s]+)', page_text)

        for actor, character in matches:
            cast_list.append(f"{actor} as {character}")

        return list(set(cast_list))

    def extract_script(self, transcript_soup):
        """Extract script text from transcript page"""
        script_text = ""

        content_div = transcript_soup.find("div", {"class": "mw-parser-output"})
        if content_div:
            paragraphs = content_div.find_all(["p", "ul", "ol"])
            for tag in paragraphs:
                script_text += self.clean_text(tag.text) + "\n"

        return script_text.strip()

    def scrape_episodes(self, urls):
        """Scrape all episodes from provided URLs"""
        self.dataset = []
        self.episode_id = 1

        for episode_url, transcript_url, cast_url in urls:
            print(f"Scraping episode {self.episode_id}: {episode_url.split('/')[-1]}...")

            try:
                ep_page = requests.get(episode_url)
                tr_page = requests.get(transcript_url)
                cast_page = requests.get(cast_url)

                episode_soup = BeautifulSoup(ep_page.content, "html.parser")
                transcript_soup = BeautifulSoup(tr_page.content, "html.parser")
                cast_soup = BeautifulSoup(cast_page.content, "html.parser")

                ep_text = episode_soup.get_text(separator='\n')
                cast_text = cast_soup.get_text(separator='\n')

                writers, directors = self.extract_writers_directors(ep_text)
                season, episode = self.extract_season_episode(ep_text)
                cast = self.extract_cast(cast_text)

                title_tag = episode_soup.find("h1", id="firstHeading")
                title = self.clean_text(title_tag.text) if title_tag else "Unknown Title"

                script = self.extract_script(transcript_soup)

                self.dataset.append({
                    "id": self.episode_id,
                    "Title": title,
                    "Season": season,
                    "Episode": episode,
                    "Writers": writers,
                    "Directors": directors,
                    "Cast": cast,
                    "Script": script
                })

                self.episode_id += 1
                time.sleep(1)  # Be respectful to the server

            except Exception as e:
                print(f"Failed scraping {episode_url}: {e}")

        return self.dataset

    def save_dataset(self, filename_base="fringe_dataset"):
        """Save dataset in multiple formats"""
        df = pd.DataFrame(self.dataset)

        # Save as CSV
        df.to_csv(f"{filename_base}.csv", index=False)
        print(f"✅ Saved dataset to {filename_base}.csv")

        # Save as JSON
        df.to_json(f"{filename_base}.json", orient="records", indent=2)
        print(f"✅ Saved dataset to {filename_base}.json")

        # Save as pickle
        with open(f'{filename_base}.pkl', 'wb') as f:
            pickle.dump(self.dataset, f)
        print(f"✅ Saved dataset to {filename_base}.pkl")

        return df

