from grids.resources.brain_structures import (
    CerebralCerebellumCortex,
    CerebralCortex,
    CerebrospinalFluidTotal,
    NeuralStructuresTotal,
    SubcorticalGreyMatter,
    TotalStructuresVolume,
    VentricularSupratentorialSystem,
    WhiteMatterCerebral,
    WhiteMatterTotal,
)


class TestBrainStructures:
    """Test cases for brain structure classes."""

    def test_cerebral_cortex(self):
        """Test CerebralCortex class."""
        cortex = CerebralCortex()

        assert cortex.cerebral_cortex_left == "Kora_mózgu_lewa"
        assert cortex.cerebral_cortex_right == "Kora_mózgu_prawa"

        # Test model_dump
        dump = cortex.model_dump()
        assert dump["cerebral_cortex_left"] == "Kora_mózgu_lewa"
        assert dump["cerebral_cortex_right"] == "Kora_mózgu_prawa"

    def test_cerebral_cerebellum_cortex(self):
        """Test CerebralCerebellumCortex class."""
        cerebellum = CerebralCerebellumCortex()

        # Inherits from CerebralCortex
        assert cerebellum.cerebral_cortex_left == "Kora_mózgu_lewa"
        assert cerebellum.cerebral_cortex_right == "Kora_mózgu_prawa"

        # Has its own fields
        assert cerebellum.cerebellum_grey_matter_left == "Istota_szara_móżdżku_lewa"
        assert cerebellum.cerebellum_grey_matter_right == "Istota_szara_móżdżku_prawa"

    def test_subcortical_grey_matter(self):
        """Test SubcorticalGreyMatter class."""
        subcortical = SubcorticalGreyMatter()

        # Test all subcortical structures
        expected_structures = {
            "thalamus_left": "Wzgórze_lewe",
            "thalamus_right": "Wzgórze_prawe",
            "caudate_nucleus_left": "Jądro_ogoniaste_lewe",
            "caudate_nucleus_right": "Jądro_ogoniaste_prawe",
            "paleostriatum_left": "Gałka_blada_lewa",
            "paleostriatum_right": "Gałka_blada_prawa",
            "amygdala_left": "Ciało_migdałowate_lewe",
            "amygdala_right": "Ciało_migdałowate_prawe",
            "nucleus_accumbens_left": "Jądro_półleżące_lewe",
            "nucleus_accumbens_right": "Jądro_półleżące_prawe",
            "putamen_left": "Skorupa_lewa",
            "putamen_right": "Skorupa_prawa",
            "hippocampus_left": "Hipokamp_lewy",
            "hippocampus_right": "Hipokamp_prawy",
        }

        for field, expected_value in expected_structures.items():
            assert getattr(subcortical, field) == expected_value

    def test_white_matter_cerebral(self):
        """Test WhiteMatterCerebral class."""
        white_matter = WhiteMatterCerebral()

        assert white_matter.white_matter_cerebral_left == "Istota_biała_lewa"
        assert white_matter.white_matter_cerebral_right == "Istota_biała_prawa"

    def test_white_matter_total(self):
        """Test WhiteMatterTotal class."""
        white_matter_total = WhiteMatterTotal()

        # Inherits from WhiteMatterCerebral
        assert white_matter_total.white_matter_cerebral_left == "Istota_biała_lewa"
        assert white_matter_total.white_matter_cerebral_right == "Istota_biała_prawa"

        # Has additional fields
        assert white_matter_total.brainstem == "Pień_mózgu"
        assert (
            white_matter_total.white_matter_cerebellum_left
            == "Istota_biała_móżdżku_lewa"
        )
        assert (
            white_matter_total.white_matter_cerebellum_right
            == "Istota_biała_móżdżku_prawa"
        )
        assert white_matter_total.midbrain_left == "Międzymózgowie_lewe"
        assert white_matter_total.midbrain_right == "Międzymózgowie_prawe"
        assert white_matter_total.optic_chiasm == "Skrzyżowanie_wzrokowe"

    def test_ventricular_supratentorial_system(self):
        """Test VentricularSupratentorialSystem class."""
        ventricular = VentricularSupratentorialSystem()

        assert ventricular.ventricular_system_left == "Układ_komorowy_mózgu_lewy"
        assert ventricular.ventricular_system_right == "Układ_komorowy_mózgu_prawy"
        assert ventricular.third_ventricle == "Komora_trzecia"

    def test_cerebrospinal_fluid_total(self):
        """Test CerebrospinalFluidTotal class."""
        csf = CerebrospinalFluidTotal()

        # Inherits from VentricularSupratentorialSystem
        assert csf.ventricular_system_left == "Układ_komorowy_mózgu_lewy"
        assert csf.ventricular_system_right == "Układ_komorowy_mózgu_prawy"
        assert csf.third_ventricle == "Komora_trzecia"

        # Has additional fields
        assert csf.cerebrospinal_fluid == "Płyn_mózgowo_rdzeniowy"
        assert csf.fourth_ventricle == "Komora_czwarta"

    def test_neural_structures_total(self):
        """Test NeuralStructuresTotal class."""
        neural = NeuralStructuresTotal()

        # Should inherit from multiple classes
        assert hasattr(neural, "cerebral_cortex_left")  # From CerebralCerebellumCortex
        assert hasattr(neural, "thalamus_left")  # From SubcorticalGreyMatter
        assert hasattr(neural, "white_matter_cerebral_left")  # From WhiteMatterTotal
        assert hasattr(neural, "brainstem")  # From WhiteMatterTotal

    def test_total_structures_volume(self):
        """Test TotalStructuresVolume class."""
        total = TotalStructuresVolume()

        # Should inherit from all relevant classes
        assert hasattr(total, "cerebral_cortex_left")  # From NeuralStructuresTotal
        assert hasattr(total, "thalamus_left")  # From NeuralStructuresTotal
        assert hasattr(
            total, "white_matter_cerebral_left"
        )  # From NeuralStructuresTotal
        assert hasattr(total, "ventricular_system_left")  # From CerebrospinalFluidTotal
        assert hasattr(total, "cerebrospinal_fluid")  # From CerebrospinalFluidTotal

    def test_structure_inheritance_hierarchy(self):
        """Test that the inheritance hierarchy is correct."""
        # Test that WhiteMatterTotal inherits from WhiteMatterCerebral
        white_matter_total = WhiteMatterTotal()
        assert isinstance(white_matter_total, WhiteMatterCerebral)

        # Test that NeuralStructuresTotal inherits from multiple classes
        neural = NeuralStructuresTotal()
        assert isinstance(neural, CerebralCerebellumCortex)
        assert isinstance(neural, SubcorticalGreyMatter)
        assert isinstance(neural, WhiteMatterTotal)

        # Test that TotalStructuresVolume inherits from both classes
        total = TotalStructuresVolume()
        assert isinstance(total, NeuralStructuresTotal)
        assert isinstance(total, CerebrospinalFluidTotal)

    def test_model_dump_functionality(self):
        """Test that all classes support model_dump method."""
        classes = [
            CerebralCortex(),
            CerebralCerebellumCortex(),
            SubcorticalGreyMatter(),
            WhiteMatterCerebral(),
            WhiteMatterTotal(),
            VentricularSupratentorialSystem(),
            CerebrospinalFluidTotal(),
            NeuralStructuresTotal(),
            TotalStructuresVolume(),
        ]

        for instance in classes:
            dump = instance.model_dump()
            assert isinstance(dump, dict)
            assert len(dump) > 0
